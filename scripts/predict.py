#!/usr/bin/env python
"""AxialTAD inference driver.

Run a trained AxialTAD model on one or more `*_25kb.npy` Hi-C patch files
and emit a single JSON file with per-file probabilities, binary predictions,
and (optionally, when labels are present) precision/recall/F1.

Usage:
    python scripts/predict.py \\
        --arch ACM --patch 15 \\
        --weights /path/to/axialtad_main.weights.h5 \\
        --input /path/to/input.npy_or_dir \\
        --output /path/to/predictions.json \\
        [--D 1] [--num-tokens 4] \\
        [--vert-only | --horiz-only] [--no-gating] \\
        [--threshold 0.5] [--batch-size 128] [--gpu 0] \\
        [--has-labels]

`--input` may be either:
  * a single .npy file, or
  * a directory containing `*_25kb.npy` files (glob matched recursively).

Each input array is expected to be the same shape produced by
`scripts/generate_samples.py`: a (N, patch*patch [+ 1]) float array. The
trailing column is a binary label when `--has-labels` is given (and only
then is it stripped before standardization).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import datetime as _dt
import json
import os
import time

import numpy as np
from tqdm import tqdm

from axialtad.preprocess import standardize_patch


def _build_kw_from_args(args) -> dict:
    """Construct the architecture-specific kwargs dict (mirrors train.py)."""
    kw = {}
    if args.arch == 'ACM':
        kw['D'] = args.D
        if args.vert_only:
            kw['vert_only'] = True
        if args.horiz_only:
            kw['horiz_only'] = True
        if args.no_gating:
            kw['no_gating'] = True
    if args.arch == 'multi_token':
        kw['num_tokens'] = args.num_tokens
    return kw


def _discover_inputs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        files = sorted(input_path.glob('*_25kb.npy'))
        if not files:
            raise SystemExit(f"no '*_25kb.npy' files in {input_path}")
        return files
    raise SystemExit(f"--input path does not exist: {input_path}")


def _load_patch_array(path: Path, patch: int, has_labels: bool):
    """Return (X, y) where y is None when --has-labels is not set.

    Accepts arrays of shape (N, patch*patch) or (N, patch*patch + 1). The
    trailing label column (when present) is stripped when --has-labels is set.
    """
    a = np.load(path)
    expected_feat = patch * patch
    if a.ndim != 2:
        raise SystemExit(f"{path}: expected 2-D array, got shape {a.shape}")
    if has_labels:
        if a.shape[1] != expected_feat + 1:
            raise SystemExit(
                f"{path}: --has-labels expects shape (N, {expected_feat + 1}); "
                f"got {a.shape}")
        feats = a[:, :-1].astype('float32')
        y = a[:, -1].astype(int)
    else:
        if a.shape[1] == expected_feat + 1:
            feats = a[:, :-1].astype('float32')
        elif a.shape[1] == expected_feat:
            feats = a.astype('float32')
        else:
            raise SystemExit(
                f"{path}: expected (N, {expected_feat}) or (N, {expected_feat + 1}); "
                f"got {a.shape}")
        y = None
    X = standardize_patch(feats, patch=patch).reshape(-1, patch, patch, 1)
    return X, y


def _metrics(preds: np.ndarray, y: np.ndarray) -> dict:
    TP = int(((preds == 1) & (y == 1)).sum())
    FP = int(((preds == 1) & (y == 0)).sum())
    FN = int(((preds == 0) & (y == 1)).sum())
    P = TP / (TP + FP) if (TP + FP) else 0.0
    R = TP / (TP + FN) if (TP + FN) else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) else 0.0
    return {'TP': TP, 'FP': FP, 'FN': FN,
            'precision': P, 'recall': R, 'f1': F1}


def main() -> int:
    ap = argparse.ArgumentParser(
        description='AxialTAD batch inference: emit per-file boundary '
                    'probabilities and binary predictions as JSON.')
    ap.add_argument('--arch', required=True,
                    choices=['ACM', 'ACF', 'ACF_removed', 'deepTAD', 'multi_token',
                             'mha_ablation', 'bilstm'],
                    help='Model architecture. "ACF" is a deprecated alias for "ACM".')
    ap.add_argument('--patch', type=int, required=True, choices=[10, 15],
                    help='Patch side length (must match training).')
    ap.add_argument('--weights', type=Path, required=True,
                    help='Path to the trained Keras weights file (.weights.h5).')
    ap.add_argument('--input', type=Path, required=True,
                    help='Either a single .npy file or a directory of '
                         '*_25kb.npy patch arrays.')
    ap.add_argument('--output', type=Path, required=True,
                    help='Output JSON path (predictions + metadata).')
    ap.add_argument('--D', type=int, default=1,
                    help='ACM.D parameter (ignored for non-ACM archs).')
    ap.add_argument('--num-tokens', type=int, default=4,
                    help='multi_token: number of tokens (ignored for others).')
    ap.add_argument('--vert-only', action='store_true',
                    help='ACM ablation: keep only vertical (row-axis) shifts.')
    ap.add_argument('--horiz-only', action='store_true',
                    help='ACM ablation: keep only horizontal (col-axis) shifts.')
    ap.add_argument('--no-gating', action='store_true',
                    help='ACM ablation: drop gate_row/gate_col multiplication.')
    ap.add_argument('--threshold', type=float, default=0.5,
                    help='Probability threshold for binarisation (default: 0.5).')
    ap.add_argument('--batch-size', type=int, default=128,
                    help='Inference batch size (default: 128).')
    ap.add_argument('--gpu', type=int, default=0,
                    help='CUDA_VISIBLE_DEVICES index (default: 0).')
    ap.add_argument('--has-labels', action='store_true',
                    help='Inputs contain a trailing label column; compute '
                         'TP/FP/FN/precision/recall/F1 per file.')
    args = ap.parse_args()

    # Normalize deprecated ACF alias → ACM (canonical external name)
    if args.arch == 'ACF':
        args.arch = 'ACM'

    # Pin GPU BEFORE importing TF (same pattern as train.py)
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    inputs = _discover_inputs(args.input)
    print(f"[predict] {len(inputs)} input file(s) from {args.input}")

    import tensorflow as tf  # noqa: F401  (imported to initialise CUDA context)
    from axialtad import models as M

    build_kw = _build_kw_from_args(args)
    print(f"[build] arch={args.arch} patch={args.patch} compile=False kw={build_kw}")
    model = M.build(
        arch=args.arch,
        patch_size=args.patch,
        compile=False,
        **build_kw,
    )
    model.load_weights(str(args.weights))
    print(f"[build] weights loaded from {args.weights}")

    results: dict[str, dict] = {}
    t0 = time.time()
    for f in tqdm(inputs, desc='predict', unit='file'):
        X, y = _load_patch_array(f, patch=args.patch, has_labels=args.has_labels)
        probs = model.predict(X, batch_size=args.batch_size, verbose=0).flatten()
        binary = (probs >= args.threshold).astype(int)
        entry = {
            'n_samples': int(probs.shape[0]),
            'probs': [float(p) for p in probs.tolist()],
            'binary': [int(b) for b in binary.tolist()],
            'n_positive': int(binary.sum()),
        }
        if args.has_labels and y is not None:
            entry['metrics'] = _metrics(binary, y)
        results[f.name] = entry

    elapsed = time.time() - t0

    out = {
        'metadata': {
            'model_arch': args.arch,
            'patch_size': args.patch,
            'threshold': args.threshold,
            'batch_size': args.batch_size,
            'weights': Path(args.weights).name,
            'input': str(args.input),
            'n_files_processed': len(inputs),
            'timestamp': _dt.datetime.now().astimezone().isoformat(),
            'elapsed_seconds': elapsed,
            'build_kw': build_kw,
        },
        'results': results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as fh:
        json.dump(out, fh, indent=2)
    print(f"[predict] wrote {args.output}  ({elapsed:.1f}s, {len(inputs)} files)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
