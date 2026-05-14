"""AxialTAD — Wilcoxon rank-sum post-hoc filter for trained models.

The filter (window_size=8, p<0.05) operates on:
  - dense Hi-C matrix (KR-balanced .txt)
  - per-bin model predictions (binary 0/1)

It (a) standardises the matrix per anti-diagonal, (b) finds "gap regions" via
zero-row windows, (c) on each non-gap process region computes a rank-sum p-value
between the diamond-shaped insulator score and the upstream/downstream triangles,
(d) keeps predicted boundaries with p<0.05, (e) deduplicates adjacent kept bins
by p-value.

Usage:
    python scripts/wilcoxon_filter.py            # process all trained models
    python scripts/wilcoxon_filter.py --model 15x15/intersection/ACM_D2  # one model

Output: results_filtered.json next to each model.weights.h5, with the same shape
as results.json plus the post-filter TP/FP/FN/P/R/F1.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import os
import re
import time

import numpy as np
from scipy.stats import ranksums

_ENV_ROOT = os.environ.get("AXIALTAD_ROOT")

# Module-level paths are populated by main() once --root / $AXIALTAD_ROOT
# is resolved. Per-subdir overrides via $WILCOXON_MATRIX_DIR / $WILCOXON_SAMPLES_ROOT
# are still honoured.
ROOT: Path | None = None
MATRIX_DIR: Path | None = None
SAMPLES_ROOT: Path | None = None
TRAINED_ROOT: Path | None = None


def _set_paths(root: Path) -> None:
    """Populate ROOT-derived module globals from a resolved working root."""
    global ROOT, MATRIX_DIR, SAMPLES_ROOT, TRAINED_ROOT
    ROOT = root
    MATRIX_DIR = Path(os.environ.get('WILCOXON_MATRIX_DIR', str(ROOT / 'matrix_kr_25kb')))
    SAMPLES_ROOT = Path(os.environ.get('WILCOXON_SAMPLES_ROOT', str(ROOT / 'samples')))
    TRAINED_ROOT = ROOT / 'trained_models'


# =============================================================================
# Student's filter_predictions logic (verbatim, refactored to take arrays)
# =============================================================================
def find_gap_regions(matrix: np.ndarray, window_size: int) -> np.ndarray:
    n = matrix.shape[0]
    gap = np.zeros(n)
    for i in range(n):
        s = max(0, i - window_size)
        e = min(n, i + window_size + 1)
        if np.sum(matrix[i, s:e]) == 0:
            gap[i] = -0.5
    return np.where(gap == -0.5)[0]


def which_process_regions(rmv_idx: np.ndarray, n_bins: int, min_size: int = 3):
    proc = sorted(set(range(n_bins)) - set(int(x) for x in rmv_idx))
    if not proc:
        return []
    regions = []
    start = proc[0]
    for i in range(1, len(proc)):
        if proc[i] - proc[i - 1] > 1:
            if proc[i - 1] - start + 1 >= min_size:
                regions.append((start, proc[i - 1]))
            start = proc[i]
    if proc[-1] - start + 1 >= min_size:
        regions.append((start, proc[-1]))
    return regions


def get_upstream_triangle(matrix, i, size):
    low = max(0, i - size)
    tri = matrix[low:i + 1, low:i + 1]
    return tri[np.triu_indices(tri.shape[0], k=1)]


def get_downstream_triangle(matrix, i, size):
    n = matrix.shape[0]
    if i >= n - 1:
        return np.array([])
    up = min(n, i + size + 1)
    tri = matrix[i + 1:up, i + 1:up]
    return tri[np.triu_indices(tri.shape[0], k=1)]


def get_diamond_matrix(matrix, i, size):
    n = matrix.shape[0]
    new = np.full((size, size), np.nan)
    for k in range(size):
        row_idx = i - k
        if row_idx < 0 or i >= n:
            continue
        l = min(i + 1, n)
        u = min(i + size + 1, n)
        new[size - k - 1, :u - l] = matrix[row_idx, l:u]
    return new.flatten()


def compute_pvalues(matrix, region, size):
    pvalues = np.ones(matrix.shape[0])
    for i in range(region[0], region[1] + 1):
        diamond = get_diamond_matrix(matrix, i, size)
        upstream = get_upstream_triangle(matrix, i, size)
        downstream = get_downstream_triangle(matrix, i, size)
        compare = np.concatenate([upstream, downstream])
        if len(diamond) == 0 or len(compare) == 0:
            continue
        valid = diamond[~np.isnan(diamond)]
        if len(valid) == 0 or len(compare) == 0:
            continue
        _, p = ranksums(valid, compare)
        pvalues[i] = p
    return pvalues


def apply_filter(matrix: np.ndarray, predicted_bins: np.ndarray,
                 window_size: int = 8, alpha: float = 0.05) -> np.ndarray:
    """Replicates student's filter_predictions; returns filtered bin indices."""
    n_bins = matrix.shape[0]

    local_ext = np.zeros(n_bins)
    if predicted_bins.size > 0:
        valid = predicted_bins[(predicted_bins >= 0) & (predicted_bins < n_bins)]
        local_ext[valid] = -1

    gap_idx = find_gap_regions(matrix, window_size)
    proc_regions = which_process_regions(gap_idx, n_bins)

    scaled = matrix.copy()
    for d in range(1, 2 * window_size + 1):
        row_idx = np.arange(matrix.shape[0] - d)
        col_idx = row_idx + d
        values = scaled[row_idx, col_idx]
        mean = np.mean(values)
        std = np.std(values) + 1e-10
        scaled[row_idx, col_idx] = (values - mean) / std

    pvalues = np.ones(n_bins)
    for region in proc_regions:
        p = compute_pvalues(scaled, region, window_size)
        pvalues[region[0]:region[1] + 1] = p[region[0]:region[1] + 1]

    final_ext = local_ext.copy()
    for i in range(n_bins):
        if local_ext[i] == -1 and pvalues[i] < alpha:
            final_ext[i] = -1
        elif local_ext[i] == -1:
            final_ext[i] = 0

    now_local = np.where(final_ext == -1)[0]
    if len(now_local) >= 2:
        filtered = []
        skip = False
        for i in range(len(now_local) - 1):
            if skip:
                skip = False
                continue
            if now_local[i] + 1 == now_local[i + 1]:
                keep = now_local[i] if pvalues[now_local[i]] < pvalues[now_local[i + 1]] else now_local[i + 1]
                filtered.append(keep)
                skip = True
            else:
                filtered.append(now_local[i])
        if not skip:
            filtered.append(now_local[-1])
        now_local = sorted(set(filtered))

    return np.array(sorted(set(int(x) for x in now_local)), dtype=int)


# =============================================================================
# Per-model processing
# =============================================================================
def process_model(model_dir: Path, gpu: int = 0) -> dict:
    """Run inference + Wilcoxon filter on the 8 test chromosomes for one model."""
    config = json.load(open(model_dir / 'config.json'))
    arch = config['arch']; patch = config['patch']; ref = config['ref']
    # Accept legacy 'ACF' arch string from older config.json files
    if arch == 'ACF':
        arch = 'ACM'
    D = config.get('D', 2); num_tokens = config.get('num_tokens', 1)
    weights = model_dir / 'model.weights.h5'

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

    from axialtad import models as M
    from axialtad.preprocess import standardize_patch
    kw = {}
    if arch == 'ACM':
        kw['D'] = D
        if config.get('vert_only'): kw['vert_only'] = True
        if config.get('horiz_only'): kw['horiz_only'] = True
        if config.get('no_gating'): kw['no_gating'] = True
    if arch == 'multi_token':
        kw['num_tokens'] = num_tokens
    model = M.build(arch, patch_size=patch, compile=False, **kw)
    model.load_weights(weights)

    test_dir = SAMPLES_ROOT / f'{patch}x{patch}' / ref / 'test'
    test_files = sorted(test_dir.glob('*_25kb.npy'))

    pre_TP = pre_FP = pre_FN = 0
    post_TP = post_FP = post_FN = 0
    per_chr = []
    for f in test_files:
        base = f.stem.replace('_25kb', '')
        a = np.load(f)
        X = standardize_patch(a[:, :-1], patch=patch).reshape(-1, patch, patch, 1)
        y = a[:, -1].astype(int)
        preds = (model.predict(X, batch_size=128, verbose=0).flatten() >= 0.5).astype(int)
        pred_bins = np.where(preds == 1)[0]
        true_bins = set(int(b) for b in np.where(y == 1)[0])

        # PRE
        ptp = sum(1 for b in pred_bins if b in true_bins)
        pfp = len(pred_bins) - ptp
        pfn = len(true_bins) - ptp
        pre_TP += ptp; pre_FP += pfp; pre_FN += pfn

        # Wilcoxon filter
        matrix_path = MATRIX_DIR / f'{base}_25kb.txt'
        matrix = np.loadtxt(matrix_path, dtype=np.float32)
        t0 = time.time()
        filt_bins = apply_filter(matrix, pred_bins, window_size=8, alpha=0.05)
        wall = time.time() - t0
        filt_set = set(int(b) for b in filt_bins)

        # POST
        ftp = len(filt_set & true_bins)
        ffp = len(filt_set - true_bins)
        ffn = len(true_bins - filt_set)
        post_TP += ftp; post_FP += ffp; post_FN += ffn

        per_chr.append({
            'file': f.name, 'true_pos': len(true_bins),
            'pre': {'pred_pos': int(len(pred_bins)), 'tp': ptp, 'fp': pfp, 'fn': pfn},
            'post': {'pred_pos': int(len(filt_set)), 'tp': ftp, 'fp': ffp, 'fn': ffn},
            'filter_wall_s': wall,
        })

    def metrics(TP, FP, FN):
        P = TP / (TP + FP) if TP + FP else 0.0
        R = TP / (TP + FN) if TP + FN else 0.0
        F1 = 2 * P * R / (P + R) if P + R else 0.0
        return {'TP': TP, 'FP': FP, 'FN': FN, 'precision': P, 'recall': R, 'f1': F1}

    out = {
        'arch': arch, 'D': D, 'num_tokens': num_tokens, 'patch': patch, 'ref': ref,
        'pre_filter': metrics(pre_TP, pre_FP, pre_FN),
        'post_filter': metrics(post_TP, post_FP, post_FN),
        'per_chr': per_chr,
    }
    with open(model_dir / 'results_filtered.json', 'w') as f:
        json.dump(out, f, indent=2)
    return out


def discover_models() -> list[Path]:
    """All trained model dirs under trained_models/."""
    out = []
    for p in sorted(TRAINED_ROOT.rglob('config.json')):
        out.append(p.parent)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--root',
        type=Path,
        default=Path(_ENV_ROOT) if _ENV_ROOT else None,
        help="Working root directory containing samples/, ontad_output/, "
             "matrix_kr_25kb/, trained_models/, etc. Defaults to "
             "$AXIALTAD_ROOT environment variable.",
    )
    ap.add_argument('--model', help='Process only this model (relative to trained_models/, e.g., 15x15/intersection/ACM_D2)')
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()
    if args.root is None:
        ap.error("--root is required (or set AXIALTAD_ROOT environment variable)")
    _set_paths(args.root)

    if args.model:
        targets = [TRAINED_ROOT / args.model]
    else:
        targets = discover_models()
    print(f"Processing {len(targets)} model(s)")

    summary = []
    for d in targets:
        t0 = time.time()
        print(f"\n[{d.relative_to(TRAINED_ROOT)}] start")
        out = process_model(d, gpu=args.gpu)
        elapsed = time.time() - t0
        pre = out['pre_filter']; post = out['post_filter']
        print(f"  pre  : TP={pre['TP']:>5d} FP={pre['FP']:>5d} FN={pre['FN']:>5d}  "
              f"P={pre['precision']:.3f} R={pre['recall']:.3f} F1={pre['f1']:.3f}")
        print(f"  post : TP={post['TP']:>5d} FP={post['FP']:>5d} FN={post['FN']:>5d}  "
              f"P={post['precision']:.3f} R={post['recall']:.3f} F1={post['f1']:.3f}")
        print(f"  wall : {elapsed:.1f}s")
        summary.append({'model': str(d.relative_to(TRAINED_ROOT)),
                        'pre_f1': pre['f1'], 'post_f1': post['f1'],
                        'wall_s': elapsed})

    print('\n=== summary ===')
    print(f"{'model':55s} {'pre_F1':>7s} {'post_F1':>8s} {'Δ':>7s} {'wall':>8s}")
    for s in summary:
        delta = s['post_f1'] - s['pre_f1']
        print(f"{s['model']:55s} {s['pre_f1']:>7.3f} {s['post_f1']:>8.3f} {delta:>+7.3f} {s['wall_s']:>7.1f}s")

if __name__ == '__main__':
    sys.exit(main())
