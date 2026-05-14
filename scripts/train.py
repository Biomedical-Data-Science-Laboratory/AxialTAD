"""AxialTAD trainer driver.

Usage:
    python scripts/train.py --arch ACM --D 2 --patch 15 --ref filtered_ontad \
        --output trained_models/15x15/filtered_ontad/ACM_D2 --gpu 1

Loads `*_processed.npy` from `samples/{patch_dir}/{ref}/train/`, trains for 30 epochs
at batch_size=128, saves model weights + history + config.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import os
import time

import numpy as np

from axialtad.preprocess import standardize_patch


# Bundled validation set (size-stratified, deterministic; see paper Methods).
VAL_SPEC_DEFAULT = {
    'GM12878': ['chr4', 'chr11', 'chr16', 'chr19'],
    'HeLa-S3': ['chr3', 'chr8',  'chr14', 'chr19'],
    'IMR-90':  ['chr3', 'chr9',  'chr14', 'chr19'],
    'K562':    ['chr4', 'chr12', 'chr19'],
}


def parse_val_spec(s):
    """Parse '--val-chrs' arg.
    'default'          → bundled validation spec
    'cell:chr1,chr2;…' → custom spec (e.g., for LOCO where one cell is held out)
    None / empty       → no validation set"""
    if not s:
        return None
    if s == 'default':
        return dict(VAL_SPEC_DEFAULT)
    out = {}
    for entry in s.split(';'):
        if not entry: continue
        cell, chrs = entry.split(':')
        out[cell.strip()] = [c.strip() for c in chrs.split(',') if c.strip()]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', required=True,
                    choices=['ACM', 'ACF', 'ACF_removed', 'deepTAD', 'multi_token',
                             'mha_ablation', 'bilstm'],
                    help='Model architecture. "ACF" is a deprecated alias for "ACM".')
    ap.add_argument('--D', type=int, default=2, help='ACM.D parameter')
    ap.add_argument('--num-tokens', type=int, default=1, help='multi_token tokens')
    ap.add_argument('--vert-only', action='store_true',
                    help='ACM: keep only vertical (row-axis) shifts')
    ap.add_argument('--horiz-only', action='store_true',
                    help='ACM: keep only horizontal (col-axis) shifts')
    ap.add_argument('--no-gating', action='store_true',
                    help='ACM: drop gate_row/gate_col multiplication')
    ap.add_argument('--patch', type=int, required=True, choices=[10, 15])
    ap.add_argument('--ref', required=True,
                    choices=['filtered_ontad', 'unfiltered_ontad', 'intersection'])
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--seed', type=int, default=123,
                    help='Sets both tf.random.set_seed and np.random.seed')
    ap.add_argument('--regen-from-raw', action='store_true',
                    help='Build training set in memory from raw .npy with seeded postprocess')
    ap.add_argument('--no-augment', action='store_true',
                    help='Skip 90° rotation augmentation in --regen-from-raw (positives kept as-is)')
    ap.add_argument('--neg-cap-mode', default='after', choices=['after', 'match-aug'],
                    help='Neg cap rule for --regen-from-raw: '
                         '"after" = 4× num_pos_after_aug (default, student spec); '
                         '"match-aug" = 8× original_pos (matches with-aug neg count when --no-augment)')
    ap.add_argument('--val-chrs', default=None,
                    help='Validation set spec. "default" = bundled validation spec; '
                         '"cell:chr1,chr2;cell:..." = custom; None = no val set.')
    _env_root = os.environ.get("AXIALTAD_ROOT")
    _samples_default = Path(_env_root) / 'samples' if _env_root else None
    ap.add_argument('--samples-root', type=Path,
                    default=_samples_default,
                    help='Directory containing training samples. '
                         'Defaults to $AXIALTAD_ROOT/samples if AXIALTAD_ROOT is set.')
    ap.add_argument('--force-train', action='store_true',
                    help='Re-train even if model.weights.h5 exists')
    ap.add_argument('--no-test-eval', action='store_true',
                    help='Skip post-training test inference')
    ap.add_argument('--exclude-cells', default=None,
                    help='Comma-separated cell list to remove from train+val (LOCO setup). '
                         'Test set is unaffected here; the runner provides a separate held-out test path.')
    ap.add_argument('--test-cells-only', default=None,
                    help='Comma-separated cell list — only evaluate test files for these cells. '
                         'Used by LOCO to test only on the held-out cell.')
    args = ap.parse_args()

    if args.samples_root is None:
        ap.error("--samples-root is required (or set AXIALTAD_ROOT environment variable)")

    # Normalize deprecated ACF alias → ACM
    if args.arch == 'ACF':
        args.arch = 'ACM'

    # Pin GPU before importing TF
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Set both numpy + python seeds for deterministic data prep
    np.random.seed(args.seed)
    import random
    random.seed(args.seed)

    # Load data first (CPU only)
    patch_dir = '15x15' if args.patch == 15 else '10x10'
    train_dir = args.samples_root / patch_dir / args.ref / 'train'
    if not train_dir.exists():
        raise SystemExit(f"train dir not found: {train_dir}")

    # LOCO: parse cell exclusion sets
    exclude_cells = set()
    if args.exclude_cells:
        exclude_cells = {c.strip() for c in args.exclude_cells.split(',') if c.strip()}
        print(f"[exclude-cells] removing from train+val: {sorted(exclude_cells)}")

    def _file_in_excluded_cell(name: str) -> bool:
        for cell in exclude_cells:
            if name.startswith(f"{cell}_"):
                return True
        return False

    # Validation set detection priority:
    #   1) If sibling val/ dir exists under {samples-root}/{patch}/{ref}/val/, auto-use it.
    #   2) Else, fall back to --val-chrs CLI spec (overrides the auto path).
    val_dir = args.samples_root / patch_dir / args.ref / 'val'
    val_dir_auto = val_dir.exists() and any(val_dir.glob('*_25kb.npy'))
    val_spec = None
    val_exclude = set()
    if val_dir_auto and not args.val_chrs:
        val_files_raw = sorted(val_dir.glob('*_25kb.npy'))
        val_files_raw = [p for p in val_files_raw if not p.name.endswith('_processed.npy')
                         and not _file_in_excluded_cell(p.name)]
        print(f"[val] auto-detected: {len(val_files_raw)} val files in {val_dir}"
              + (f" (excluded cells: {sorted(exclude_cells)})" if exclude_cells else ''))
        val_spec = {'__auto__': [p.name for p in val_files_raw]}
    else:
        val_spec_parsed = parse_val_spec(args.val_chrs)
        if val_spec_parsed is not None:
            val_spec = val_spec_parsed
            for cell, chrs in val_spec.items():
                for c in chrs:
                    val_exclude.add(f"{cell}_{c}_25kb.npy")
                    val_exclude.add(f"{cell}_{c}_25kb_processed.npy")
            print(f"[val] excluding {sum(len(v) for v in val_spec.values())} chrs from train: {val_spec}")

    if args.regen_from_raw:
        raw_files = sorted(p for p in train_dir.glob('*_25kb.npy')
                           if not p.name.endswith('_processed.npy')
                           and p.name not in val_exclude
                           and not _file_in_excluded_cell(p.name))
        if not raw_files:
            raise SystemExit(f"no raw *_25kb.npy in {train_dir}")
        print(f"[load] regen_from_raw: {len(raw_files)} raw train files (seed={args.seed})")
        arrs = []
        for f in raw_files:
            raw = np.load(f)
            pos = raw[raw[:, -1] == 1]
            neg = raw[raw[:, -1] == 0]
            if args.no_augment:
                rotated = np.empty((0, raw.shape[1]))
            else:
                rotated = []
                for s in pos:
                    m = s[:-1].reshape(args.patch, args.patch)
                    r = np.rot90(m, k=-1).flatten()
                    rotated.append(np.append(r, 1))
                rotated = np.array(rotated) if rotated else np.empty((0, raw.shape[1]))
            num_pos = len(pos) + len(rotated)
            if args.neg_cap_mode == 'match-aug':
                neg_limit = min(len(neg), len(pos) * 8)
            else:
                neg_limit = min(len(neg), num_pos * 4)
            sampled_neg = neg[np.random.choice(len(neg), size=neg_limit, replace=False)] if neg_limit > 0 else neg[:0]
            parts = [pos]
            if len(rotated) > 0: parts.append(rotated)
            parts.append(sampled_neg)
            final = np.vstack(parts)
            np.random.shuffle(final)
            arrs.append(final)
        train_data = np.vstack(arrs)
        train_files_list = [p.name for p in raw_files]
    else:
        proc_files = sorted(p for p in train_dir.glob('*_processed.npy')
                            if p.name not in val_exclude
                            and not _file_in_excluded_cell(p.name))
        if not proc_files:
            raise SystemExit(f"no *_processed.npy files in {train_dir}")
        print(f"[load] {len(proc_files)} processed train files in {train_dir}")
        arrs = [np.load(p) for p in proc_files]
        train_data = np.vstack(arrs)
        train_files_list = [p.name for p in proc_files]
    print(f"[load] train_data shape: {train_data.shape}")

    X_flat = train_data[:, :-1].astype('float32')
    X_flat = standardize_patch(X_flat, patch=args.patch)
    X = X_flat.reshape(-1, args.patch, args.patch, 1)
    y = train_data[:, -1].astype('float32')

    print(f"[load] X={X.shape} y={y.shape} pos={int(y.sum())} neg={int((y==0).sum())}")

    # Load validation set (raw, test-style preprocessing — no rotation augmentation)
    X_val = y_val = None
    val_files_list = []
    if val_spec is not None:
        v_arrs_X = []; v_arrs_y = []
        if '__auto__' in val_spec:
            # Auto-detected from sibling val/ dir
            for fname in val_spec['__auto__']:
                f = val_dir / fname
                if not f.exists():
                    raise SystemExit(f"val file missing: {f}")
                a = np.load(f)
                Xv = standardize_patch(a[:, :-1], patch=args.patch).reshape(-1, args.patch, args.patch, 1)
                yv = a[:, -1].astype('float32')
                v_arrs_X.append(Xv); v_arrs_y.append(yv)
                val_files_list.append(f.name)
        else:
            # CLI-specified per-cell chr list (legacy path)
            for cell, chrs in val_spec.items():
                for c in chrs:
                    f = train_dir / f"{cell}_{c}_25kb.npy"
                    if not f.exists():
                        raise SystemExit(f"val file missing: {f}")
                    a = np.load(f)
                    Xv = standardize_patch(a[:, :-1], patch=args.patch).reshape(-1, args.patch, args.patch, 1)
                    yv = a[:, -1].astype('float32')
                    v_arrs_X.append(Xv); v_arrs_y.append(yv)
                    val_files_list.append(f.name)
        X_val = np.concatenate(v_arrs_X)
        y_val = np.concatenate(v_arrs_y)
        print(f"[val] X_val={X_val.shape} y_val={y_val.shape} "
              f"pos={int(y_val.sum())} neg={int((y_val == 0).sum())}")

    # Now import TF (with GPU pinned)
    import tensorflow as tf
    from axialtad import models as M

    gpus = tf.config.list_physical_devices('GPU')
    print(f"[TF] visible GPUs: {len(gpus)} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")

    # Build model
    kw = {}
    if args.arch == 'ACM':
        kw['D'] = args.D
        if args.vert_only: kw['vert_only'] = True
        if args.horiz_only: kw['horiz_only'] = True
        if args.no_gating: kw['no_gating'] = True
    if args.arch == 'multi_token':
        kw['num_tokens'] = args.num_tokens
    model = M.build(args.arch, patch_size=args.patch, lr=args.lr, seed=args.seed, **kw)
    print(f"[build] arch={args.arch} patch={args.patch} lr={args.lr} seed={args.seed} kw={kw}")
    model.summary(print_fn=lambda s: print('[summary] ' + s))

    args.output.mkdir(parents=True, exist_ok=True)

    # Save config
    config = {
        'arch': args.arch, 'D': args.D, 'num_tokens': args.num_tokens,
        'vert_only': args.vert_only, 'horiz_only': args.horiz_only, 'no_gating': args.no_gating,
        'patch': args.patch, 'ref': args.ref,
        'epochs': args.epochs, 'batch_size': args.batch_size,
        'optimizer': 'Adam', 'learning_rate': args.lr,
        'loss': 'BinaryCrossentropy(label_smoothing=0.01)',
        'tf_seed': args.seed,
        'samples_root': str(args.samples_root),
        'train_files': train_files_list,
        'regen_from_raw': args.regen_from_raw,
        'no_augment': args.no_augment,
        'neg_cap_mode': args.neg_cap_mode,
        'val_chrs_arg': args.val_chrs,
        'val_spec': val_spec,
        'val_files': val_files_list,
        'X_shape': list(X.shape), 'pos': int(y.sum()), 'neg': int((y == 0).sum()),
    }
    with open(args.output / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    weights_path = args.output / 'model.weights.h5'
    history_path = args.output / 'history.json'

    # Per-epoch validation F1 callback; saves best-val-epoch weights.
    class ValF1Callback(tf.keras.callbacks.Callback):
        def __init__(self, X_val, y_val, weights_path, batch_size=256):
            super().__init__()
            self.X_val = X_val; self.y_val = y_val
            self.weights_path = str(weights_path)
            self.batch_size = batch_size
            self.best_f1 = -1.0
            self.best_epoch = -1
            self.history = []

        def on_epoch_end(self, epoch, logs=None):
            preds = (self.model.predict(self.X_val, batch_size=self.batch_size, verbose=0)
                     .flatten() >= 0.5).astype(int)
            yv = self.y_val.astype(int)
            TP = int(((preds == 1) & (yv == 1)).sum())
            FP = int(((preds == 1) & (yv == 0)).sum())
            FN = int(((preds == 0) & (yv == 1)).sum())
            P = TP / (TP + FP) if TP + FP else 0.0
            R = TP / (TP + FN) if TP + FN else 0.0
            F1 = 2 * P * R / (P + R) if P + R else 0.0
            (logs or {})['val_f1'] = F1
            self.history.append({'epoch': epoch + 1, 'val_f1': F1,
                                 'val_TP': TP, 'val_FP': FP, 'val_FN': FN,
                                 'val_P': P, 'val_R': R})
            tag = ''
            if F1 > self.best_f1:
                self.best_f1 = F1
                self.best_epoch = epoch + 1
                self.model.save_weights(self.weights_path)
                tag = ' ↑ best (saved)'
            print(f"  [val] epoch {epoch+1}: F1={F1:.4f}{tag}")

    # Train (or resume from existing weights)
    elapsed = 0.0
    if weights_path.exists() and history_path.exists() and not args.force_train:
        print(f"[skip-train] using existing {weights_path}")
        model.load_weights(weights_path)
        with open(history_path) as f:
            hist = json.load(f)
        elapsed = hist.get('elapsed_seconds', 0.0)
        last = {k: vs[-1] for k, vs in hist.items()
                if isinstance(vs, list) and vs}
    else:
        t0 = time.time()
        print(f"[train] starting fit at {time.strftime('%H:%M:%S')}")
        callbacks = []
        val_cb = None
        if X_val is not None:
            val_cb = ValF1Callback(X_val, y_val, weights_path)
            callbacks.append(val_cb)
        history = model.fit(X, y, epochs=args.epochs, batch_size=args.batch_size,
                            verbose=2, callbacks=callbacks)
        elapsed = time.time() - t0
        print(f"[train] done in {elapsed:.1f}s")
        if val_cb is not None:
            # callback already saved best-val-epoch weights to weights_path
            print(f"[val] best epoch={val_cb.best_epoch}  best val_F1={val_cb.best_f1:.4f}")
            # restore best-val weights into the live model (callback may have left final-epoch weights in memory)
            model.load_weights(weights_path)
        else:
            # No val: save final-epoch weights (legacy behaviour)
            model.save_weights(weights_path)
            print(f"[save] {weights_path}")
        hist = {k: [float(v) for v in vs] for k, vs in history.history.items()}
        hist['elapsed_seconds'] = elapsed
        if val_cb is not None:
            hist['val_history'] = val_cb.history
            hist['best_val_epoch'] = val_cb.best_epoch
            hist['best_val_f1'] = val_cb.best_f1
        with open(history_path, 'w') as f:
            json.dump(hist, f, indent=2)
        last = {k: vs[-1] for k, vs in hist.items()
                if isinstance(vs, list) and vs}
    print(f"[final-train] {last}")

    # Post-training test inference
    if not args.no_test_eval:
        test_dir = args.samples_root / patch_dir / args.ref / 'test'
        test_files = sorted(test_dir.glob('*_25kb.npy'))
        if args.test_cells_only:
            test_cells = {c.strip() for c in args.test_cells_only.split(',') if c.strip()}
            test_files = [p for p in test_files
                          if any(p.name.startswith(f"{c}_") for c in test_cells)]
            print(f"[test] limiting to cells {sorted(test_cells)}: {len(test_files)} files")
        TP = FP = FN = 0
        per_chr = []
        for f in test_files:
            a = np.load(f)
            Xt = standardize_patch(a[:, :-1], patch=args.patch).reshape(-1, args.patch, args.patch, 1)
            yt = a[:, -1].astype(int)
            preds = (model.predict(Xt, batch_size=128, verbose=0).flatten() >= 0.5).astype(int)
            tp = int(((preds == 1) & (yt == 1)).sum())
            fp = int(((preds == 1) & (yt == 0)).sum())
            fn = int(((preds == 0) & (yt == 1)).sum())
            TP += tp; FP += fp; FN += fn
            per_chr.append({'file': f.name, 'samples': int(a.shape[0]),
                            'pos': int(yt.sum()), 'tp': tp, 'fp': fp, 'fn': fn})
        P = TP / (TP + FP) if TP + FP else 0.0
        R = TP / (TP + FN) if TP + FN else 0.0
        F1 = 2 * P * R / (P + R) if P + R else 0.0
        results = {
            'arch': args.arch, 'D': args.D, 'num_tokens': args.num_tokens,
            'patch': args.patch, 'ref': args.ref,
            'wall_seconds': elapsed,
            'train_final': last,
            'test_aggregated': {'TP': TP, 'FP': FP, 'FN': FN,
                                'precision': P, 'recall': R, 'f1': F1},
            'test_per_chr': per_chr,
        }
        with open(args.output / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"[test-eval] TP={TP} FP={FP} FN={FN}  P={P:.4f} R={R:.4f} F1={F1:.4f}")
        # Sanity FLAGS
        if F1 < 0.05:
            print(f"[FLAG] F1<0.05 — possible failed training")
        elif F1 > 0.95:
            print(f"[FLAG] F1>0.95 — possible data leakage")
    return 0


if __name__ == '__main__':
    sys.exit(main())
