"""AxialTAD — model factories.

5 architectures, single factory `build(arch, patch_size, **kwargs)`:
  - 'ACM'           — Axial Contrast Module model (param: D ∈ {1, 2}).
                      'ACF' is accepted as a deprecated alias.
  - 'deepTAD'       — deepTAD repo Cell 3 verbatim, parameterised by patch_size
  - 'multi_token'   — deepTAD architecture with adjustable num_tokens (4|25|100)
  - 'mha_ablation'  — deepTAD baseline with MHA → per-token Dense
  - 'bilstm'        — direct row-wise Bi-LSTM(128) on (10×10) input

Common compile config:
    loss = BinaryCrossentropy(label_smoothing=0.01)
    optimizer = Adam(learning_rate=3e-4)
    metrics = [tp, fp, tn, fn, accuracy, precision, recall]
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.layers import multiply


# =============================================================================
# Common compile
# =============================================================================
def _compile(model: tf.keras.Model, lr: float = 3e-4) -> tf.keras.Model:
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.01),
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        metrics=[
            keras.metrics.TruePositives(name='tp'),
            keras.metrics.FalsePositives(name='fp'),
            keras.metrics.TrueNegatives(name='tn'),
            keras.metrics.FalseNegatives(name='fn'),
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
        ],
    )
    return model


# =============================================================================
# CBAM (channel + spatial) — used by deepTAD + variants
# =============================================================================
def channel_attention(input_feature, ratio=8):
    channel = input_feature.shape[-1]
    filters = max(1, int(channel // ratio))
    shared_one = layers.Dense(filters, activation='relu',
                              kernel_initializer='he_normal',
                              use_bias=True, bias_initializer='zeros')
    shared_two = layers.Dense(channel, kernel_initializer='he_normal',
                              use_bias=True, bias_initializer='zeros')

    avg_pool = layers.GlobalAveragePooling2D()(input_feature)
    avg_pool = layers.Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_one(avg_pool)
    avg_pool = shared_two(avg_pool)

    max_pool = layers.GlobalMaxPooling2D()(input_feature)
    max_pool = layers.Reshape((1, 1, channel))(max_pool)
    max_pool = shared_one(max_pool)
    max_pool = shared_two(max_pool)

    feat = layers.Add()([avg_pool, max_pool])
    feat = layers.Activation('sigmoid')(feat)
    return multiply([input_feature, feat])


def spatial_attention(input_feature, kernel_size=(3, 3)):
    avg_pool = layers.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(input_feature)
    max_pool = layers.Lambda(lambda x: K.max(x, axis=3, keepdims=True))(input_feature)
    concat = layers.Concatenate(axis=3)([avg_pool, max_pool])
    feat = layers.Conv2D(filters=1, kernel_size=kernel_size, strides=1,
                         padding='same', activation='sigmoid',
                         kernel_initializer='he_normal', use_bias=False)(concat)
    return multiply([input_feature, feat])


def cbam_block(x, ratio=8, kernel_size=(3, 3)):
    """deepTAD's full CBAM (channel + spatial)."""
    x = channel_attention(x, ratio)
    x = spatial_attention(x, kernel_size)
    return x


def cbam_channel_only(x, ratio=8):
    """ACM uses only the channel-attention branch of CBAM."""
    return channel_attention(x, ratio)


# =============================================================================
# ACM layer — Axial Contrast Module
# =============================================================================
class ACF(layers.Layer):
    """Axial Contrast Module (ACM).

    Note: The class name is retained as `ACF` for checkpoint
    compatibility with pretrained weights. The model is referred
    to as ACM in the AxialTAD paper and documentation.
    """

    def __init__(self, D=1, mid_ch=32, k_row=5, k_col=5, return_center=True,
                 vert_only=False, horiz_only=False, no_gating=False, **kwargs):
        super().__init__(**kwargs)
        self.D = D
        self.mid_ch = mid_ch
        self.k_row = k_row
        self.k_col = k_col
        self.return_center = return_center
        self.vert_only = vert_only
        self.horiz_only = horiz_only
        self.no_gating = no_gating
        if vert_only and horiz_only:
            raise ValueError('vert_only and horiz_only are mutually exclusive')

        if not horiz_only:
            self.row_conv = keras.Sequential([
                layers.SeparableConv2D(self.mid_ch, kernel_size=(3, self.k_row),
                                       padding='same', activation='relu'),
                layers.Conv2D(self.mid_ch, kernel_size=1, activation='relu'),
            ])
            if not no_gating:
                self.gate_row = keras.Sequential([layers.Conv2D(1, 1, activation='sigmoid')])
        if not vert_only:
            self.col_conv = keras.Sequential([
                layers.SeparableConv2D(self.mid_ch, kernel_size=(self.k_col, 3),
                                       padding='same', activation='relu'),
                layers.Conv2D(self.mid_ch, kernel_size=1, activation='relu'),
            ])
            if not no_gating:
                self.gate_col = keras.Sequential([layers.Conv2D(1, 1, activation='sigmoid')])
        self.head = keras.Sequential([
            layers.Conv2D(self.mid_ch, 1, activation='relu'),
            layers.Conv2D(1, 1, activation=None),
        ])

    def _shift(self, x, dy=0, dx=0):
        B, H, W, C = tf.unstack(tf.shape(x))
        pad_top   = tf.maximum( dy, 0)
        pad_bot   = tf.maximum(-dy, 0)
        pad_left  = tf.maximum( dx, 0)
        pad_right = tf.maximum(-dx, 0)
        xpad = tf.pad(x, [[0, 0], [pad_top, pad_bot], [pad_left, pad_right], [0, 0]])
        y0 = tf.maximum(-dy, 0); y1 = y0 + H
        x0 = tf.maximum(-dx, 0); x1 = x0 + W
        return xpad[:, y0:y1, x0:x1, :]

    def call(self, x, training=None):
        diffs_row = []; diffs_col = []
        for d in range(1, self.D + 1):
            if not self.horiz_only:
                down = self._shift(x, dy=+d); up = self._shift(x, dy=-d)
                dr = down - x; ur = up - x
                diffs_row += [dr, ur, tf.abs(dr), tf.abs(ur)]
            if not self.vert_only:
                right = self._shift(x, dx=+d); left = self._shift(x, dx=-d)
                rc = right - x; lc = left - x
                diffs_col += [rc, lc, tf.abs(rc), tf.abs(lc)]
        if self.horiz_only:
            Fc = tf.concat(diffs_col, axis=-1)
            C = self.col_conv(Fc)
            if not self.no_gating:
                C = C * self.gate_col(C)
            fused = C
        elif self.vert_only:
            Fr = tf.concat(diffs_row, axis=-1)
            R = self.row_conv(Fr)
            if not self.no_gating:
                R = R * self.gate_row(R)
            fused = R
        else:
            Fr = tf.concat(diffs_row, axis=-1)
            Fc = tf.concat(diffs_col, axis=-1)
            R = self.row_conv(Fr); C = self.col_conv(Fc)
            if not self.no_gating:
                R = R * self.gate_row(R); C = C * self.gate_col(C)
            fused = tf.concat([R, C], axis=-1)
        P = self.head(fused)
        if self.return_center:
            H = tf.shape(P)[1]; W = tf.shape(P)[2]
            i = H // 2; j = W // 2
            center = P[:, i:i + 1, j:j + 1, :]
            return tf.reshape(center, [tf.shape(P)[0], 1])
        return P


# =============================================================================
# Transformer encoder (deepTAD verbatim) + MHA-ablation variant
# =============================================================================
def transformer_encoder(inputs, d_model=128, num_heads=4, mlp_dim=512, dropout=0.1):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout)(x, x)
    x = layers.Add()([inputs, x])
    y = layers.LayerNormalization(epsilon=1e-6)(x)
    y = layers.Dense(mlp_dim, activation='gelu')(y)
    y = layers.Dropout(dropout)(y)
    y = layers.Dense(d_model, activation='gelu')(y)
    y = layers.Dropout(dropout)(y)
    return layers.Add()([inputs, y])  # deepTAD adds back inputs (not x), preserved


def transformer_encoder_no_mha(inputs, d_model=128, mlp_dim=512, dropout=0.1):
    """MHA → per-token Dense(d_model). FFN+residual+LN preserved."""
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.Dense(d_model)(x)              # ← MHA replaced
    x = layers.Add()([inputs, x])
    y = layers.LayerNormalization(epsilon=1e-6)(x)
    y = layers.Dense(mlp_dim, activation='gelu')(y)
    y = layers.Dropout(dropout)(y)
    y = layers.Dense(d_model, activation='gelu')(y)
    y = layers.Dropout(dropout)(y)
    return layers.Add()([inputs, y])


# =============================================================================
# Architecture factories
# =============================================================================
def init_acf(D=2, patch_size=15, vert_only=False, horiz_only=False, no_gating=False):
    """ACM-based AxialTAD classifier with D parameterised + ACM component ablation flags."""
    inp = layers.Input(shape=(patch_size, patch_size, 1))
    x = layers.Conv2D(128, 3, padding='same', activation='relu',
                      kernel_initializer='he_normal')(inp)
    x = layers.MaxPooling2D((3, 3), strides=(3, 3))(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu',
                      kernel_initializer='he_normal')(x)
    x = cbam_channel_only(x, ratio=8)
    h = ACF(D=D, mid_ch=32, k_row=5, k_col=5, return_center=True,
            vert_only=vert_only, horiz_only=horiz_only, no_gating=no_gating)(x)
    h = layers.Dense(128, activation=None, kernel_initializer='he_normal')(h)
    h = layers.PReLU()(h); h = layers.Dropout(0.4)(h)
    h = layers.Dense(64,  activation=None, kernel_initializer='he_normal')(h)
    h = layers.PReLU()(h); h = layers.Dropout(0.4)(h)
    out = layers.Dense(1, activation='sigmoid')(h)
    return _compile(tf.keras.Model(inp, out))


def init_acf_removed(patch_size=15):
    """ACM block removed. Keeps CNN(128→pool→64) + CBAM(channel) → 1×1 head → center → classifier head."""
    inp = layers.Input(shape=(patch_size, patch_size, 1))
    x = layers.Conv2D(128, 3, padding='same', activation='relu',
                      kernel_initializer='he_normal')(inp)
    x = layers.MaxPooling2D((3, 3), strides=(3, 3))(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu',
                      kernel_initializer='he_normal')(x)
    x = cbam_channel_only(x, ratio=8)
    # Replace ACM with a 1×1 conv head + center extract (matches ACM's center-pixel signature)
    h = layers.Conv2D(32, 1, activation='relu')(x)
    h = layers.Conv2D(1, 1, activation=None)(h)
    # Center pixel extract (matches ACM return_center=True path)
    H = h.shape[1]; W = h.shape[2]
    h = layers.Lambda(lambda t: t[:, H // 2:H // 2 + 1, W // 2:W // 2 + 1, :])(h)
    h = layers.Reshape((1,))(h)
    h = layers.Dense(128, activation=None, kernel_initializer='he_normal')(h)
    h = layers.PReLU()(h); h = layers.Dropout(0.4)(h)
    h = layers.Dense(64,  activation=None, kernel_initializer='he_normal')(h)
    h = layers.PReLU()(h); h = layers.Dropout(0.4)(h)
    out = layers.Dense(1, activation='sigmoid')(h)
    return _compile(tf.keras.Model(inp, out))


def init_deeptad(patch_size=10):
    """deepTAD repo Cell 3 verbatim, parameterised by patch_size."""
    inp = layers.Input(shape=(patch_size, patch_size, 1))
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(inp)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = cbam_block(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    tokens = layers.Reshape((1, 128))(x)        # 1 token
    enc = transformer_encoder(tokens, d_model=128, num_heads=4, mlp_dim=512, dropout=0.1)
    h = layers.Flatten()(enc)
    h = layers.Dense(128, activation='relu')(h); h = layers.Dropout(0.4)(h)
    h = layers.Dense(64,  activation='relu')(h); h = layers.Dropout(0.4)(h)
    out = layers.Dense(1, activation='sigmoid')(h)
    return _compile(tf.keras.Model(inp, out))


def init_multi_token(num_tokens=4, patch_size=10, d_model=128):
    """deepTAD-derived multi-token. Adjusts CNN pooling + AvgPool to coerce feature map to target token grid.

    patch_size=10 (Phase 5 Track A):
      num_tokens=4   → Conv→MaxPool→Conv→MaxPool → (2,2,64)
      num_tokens=25  → Conv→MaxPool→Conv         → (5,5,64)
      num_tokens=100 → Conv→Conv                 → (10,10,64)

    patch_size=15 (Phase 7, intersection setup):
      num_tokens=4   → Conv→MaxPool→Conv→MaxPool → (3,3,64) → AvgPool(2,1) → (2,2,64)
      num_tokens=25  → Conv→MaxPool→Conv         → (7,7,64) → AvgPool(3,1) → (5,5,64)
      num_tokens=100 → Conv→Conv                 → (15,15,64) → AvgPool(6,1) → (10,10,64)
    """
    if patch_size not in (10, 15):
        raise ValueError(f'multi_token supports patch_size 10 or 15; got {patch_size}')
    if num_tokens not in (4, 25, 100):
        raise ValueError(f'num_tokens must be 4, 25, or 100; got {num_tokens}')
    inp = layers.Input(shape=(patch_size, patch_size, 1))
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(inp)
    if num_tokens <= 25:
        x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    if num_tokens == 4:
        x = layers.MaxPooling2D((2, 2))(x)
    x = cbam_block(x)

    if patch_size == 15:
        # AvgPool spatial reduction to exact target grid (preserves channel=64)
        if num_tokens == 4:
            x = layers.AveragePooling2D(pool_size=(2, 2), strides=(1, 1))(x)   # 3→2
        elif num_tokens == 25:
            x = layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1))(x)   # 7→5
        elif num_tokens == 100:
            x = layers.AveragePooling2D(pool_size=(6, 6), strides=(1, 1))(x)   # 15→10

    if num_tokens == 4:
        tokens = layers.Reshape((4, 64))(x)
    elif num_tokens == 25:
        tokens = layers.Reshape((25, 64))(x)
    elif num_tokens == 100:
        tokens = layers.Reshape((100, 64))(x)

    tokens = layers.Dense(d_model, activation=None)(tokens)
    enc = transformer_encoder(tokens, d_model=d_model, num_heads=4, mlp_dim=512, dropout=0.1)
    h = layers.Flatten()(enc)
    h = layers.Dense(128, activation='relu')(h); h = layers.Dropout(0.4)(h)
    h = layers.Dense(64,  activation='relu')(h); h = layers.Dropout(0.4)(h)
    out = layers.Dense(1, activation='sigmoid')(h)
    return _compile(tf.keras.Model(inp, out))


def init_mha_ablation(patch_size=10):
    """deepTAD baseline with MHA → per-token Dense (1 token still)."""
    inp = layers.Input(shape=(patch_size, patch_size, 1))
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(inp)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = cbam_block(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    tokens = layers.Reshape((1, 128))(x)
    enc = transformer_encoder_no_mha(tokens, d_model=128, mlp_dim=512, dropout=0.1)
    h = layers.Flatten()(enc)
    h = layers.Dense(128, activation='relu')(h); h = layers.Dropout(0.4)(h)
    h = layers.Dense(64,  activation='relu')(h); h = layers.Dropout(0.4)(h)
    out = layers.Dense(1, activation='sigmoid')(h)
    return _compile(tf.keras.Model(inp, out))


def init_bilstm(patch_size=10, hidden=128):
    """Direct row-wise Bi-LSTM. Input (P, P, 1) → squeeze → (P, P) sequence."""
    inp = layers.Input(shape=(patch_size, patch_size, 1))
    x = layers.Reshape((patch_size, patch_size))(inp)        # (P, P)
    x = layers.Bidirectional(layers.LSTM(hidden, return_sequences=False))(x)  # (2*hidden,)
    h = layers.Dense(128, activation='relu')(x); h = layers.Dropout(0.4)(h)
    h = layers.Dense(64,  activation='relu')(h); h = layers.Dropout(0.4)(h)
    out = layers.Dense(1, activation='sigmoid')(h)
    return _compile(tf.keras.Model(inp, out))


# =============================================================================
# Factory entry point
# =============================================================================
def build(arch: str, patch_size: int = 15, lr: float = None, seed: int = None,
          compile: bool = True, **kw) -> tf.keras.Model:
    """Build an AxialTAD model.

    Args:
        arch: One of 'ACM', 'ACF_removed', 'deepTAD', 'multi_token',
            'mha_ablation', 'bilstm'. 'ACF' is accepted as a deprecated alias
            for 'ACM' (the class name is retained for checkpoint compatibility).
        patch_size: Input patch size (10 or 15).
        lr: Learning rate. Required when `compile=True`. When `compile=False`
            this argument is ignored.
        seed: If provided, sets `tf.random.set_seed(seed)` for deterministic
            weight initialisation.
        compile: When True (default), the returned model is compiled with the
            standard AxialTAD loss/optimizer/metrics. Set False for inference,
            where compilation is unnecessary and `lr` is not required.
        **kw: Architecture-specific kwargs (D, num_tokens, vert_only, ...).

    Returns:
        A (possibly compiled) `tf.keras.Model`.
    """
    if seed is not None:
        tf.random.set_seed(seed)
    if arch in ('ACM', 'ACF'):
        m = init_acf(D=kw.get('D', 2), patch_size=patch_size,
                     vert_only=kw.get('vert_only', False),
                     horiz_only=kw.get('horiz_only', False),
                     no_gating=kw.get('no_gating', False))
    elif arch == 'ACF_removed':
        m = init_acf_removed(patch_size=patch_size)
    elif arch == 'deepTAD':
        m = init_deeptad(patch_size=patch_size)
    elif arch == 'multi_token':
        m = init_multi_token(num_tokens=kw['num_tokens'], patch_size=patch_size)
    elif arch == 'mha_ablation':
        m = init_mha_ablation(patch_size=patch_size)
    elif arch == 'bilstm':
        m = init_bilstm(patch_size=patch_size, hidden=kw.get('hidden', 128))
    else:
        raise ValueError(f'unknown arch: {arch}')

    if compile:
        assert lr is not None, "lr is required when compile=True"
        # Re-compile with the requested lr (over-rides default 3e-4 baked into init_*)
        if lr != 3e-4:
            _compile(m, lr=lr)
    return m
