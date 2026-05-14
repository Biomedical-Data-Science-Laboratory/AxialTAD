#!/usr/bin/env python
"""Phase 4 (Path C) sample generation.

For each (cell line, chromosome) in the 83-chr KR-balanced set, produce 15×15 patch
samples with binary boundary labels under three references:
  1. filtered_ontad     (OnTAD level==1 AND TADscore>=2)
  2. unfiltered_ontad   (OnTAD level>=1; student's get_ontad_all_bins as-is)
  3. intersection       (unfiltered_ontad ∩ HiCExplorer hicFindTADs boundaries)

Train chromosomes (75) get an additional `_processed.npy` via training_postprocess
(positive 90° rotation augmentation + negative cap at 4× num_pos + shuffle).
Test chromosomes (8) only produce raw `.npy`.

Random sampling in training_postprocess is intentionally unseeded — matches student
methodology (manuscript Methods §3.2).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
_ENV_ROOT = os.environ.get("AXIALTAD_ROOT")

# Module-level paths are populated by main() once --root / $AXIALTAD_ROOT
# is resolved. The fine-grained --matrix-dir / --ontad-dir / --hic-dir /
# --samples-root flags continue to override individual subdirs.
ROOT: Path | None = None
MATRIX_DIR: Path | None = None
ONTAD_DIR: Path | None = None
HIC_DIR: Path | None = None
SAMPLES: Path | None = None


def _set_paths(root: Path) -> None:
    """Populate ROOT-derived module globals from a resolved working root."""
    global ROOT, MATRIX_DIR, ONTAD_DIR, HIC_DIR, SAMPLES
    ROOT = root
    MATRIX_DIR = ROOT / 'matrix_kr_25kb'
    ONTAD_DIR = ROOT / 'ontad_output'
    HIC_DIR = ROOT / 'hicexplorer_output_v2'
    SAMPLES = ROOT / 'samples'

# -----------------------------------------------------------------------------
# Data split
# -----------------------------------------------------------------------------
TEST_SET = {
    ('GM12878', 'chr1'),  ('GM12878', 'chr5'),
    ('HeLa-S3', 'chr13'), ('HeLa-S3', 'chr15'),
    ('IMR-90',  'chr7'),  ('IMR-90',  'chr11'),
    ('K562',    'chr17'), ('K562',    'chr20'),
}

REFS = ('filtered_ontad', 'unfiltered_ontad', 'intersection')

# -----------------------------------------------------------------------------
# Reference selectors
# -----------------------------------------------------------------------------
def get_ontad_bins_unfiltered(tad_file: Path) -> np.ndarray:
    """Student's get_ontad_all_bins() — skip the level-0 row, keep all level>=1."""
    df = pd.read_csv(tad_file, header=None, sep=r'\s+', comment='#')
    if len(df) == 0:
        return np.array([], dtype=int)
    df = df.iloc[1:, :]
    starts = df.iloc[:, 0].astype(int).to_numpy() - 1
    ends   = df.iloc[:, 1].astype(int).to_numpy() - 1
    return np.unique(np.concatenate([starts, ends]))


def get_ontad_bins_filtered(tad_file: Path, score_thresh: float = 2.0) -> np.ndarray:
    """level==1 AND TADscore>=score_thresh (student dissertation §2.1 / manuscript §3.1)."""
    df = pd.read_csv(tad_file, header=None, sep=r'\s+', comment='#')
    if len(df) == 0:
        return np.array([], dtype=int)
    mask = (df.iloc[:, 2] == 1) & (df.iloc[:, 4] >= score_thresh)
    df = df[mask]
    if df.empty:
        return np.array([], dtype=int)
    starts = df.iloc[:, 0].astype(int).to_numpy() - 1
    ends   = df.iloc[:, 1].astype(int).to_numpy() - 1
    return np.unique(np.concatenate([starts, ends]))


def load_hic_bins_from_bed(bed_file: Path, chrom: str, res: int = 25000,
                           use_center: bool = True) -> np.ndarray:
    """Student's load_hic_bins_from_bed."""
    if not bed_file.exists():
        return np.array([], dtype=int)
    df = pd.read_csv(bed_file, sep='\t', header=None, comment='#')
    if df.shape[1] < 3:
        raise ValueError(f"unexpected columns in {bed_file}")
    df = df[df.iloc[:, 0] == chrom].copy()
    if df.empty:
        return np.array([], dtype=int)
    start = df.iloc[:, 1].astype(np.int64).to_numpy()
    end   = df.iloc[:, 2].astype(np.int64).to_numpy()
    if use_center:
        pos = (start + end) // 2
        bins = np.rint(pos / res).astype(int)
    else:
        bins = (start // res).astype(int)
    return np.unique(bins)


def intersect_bins(a: np.ndarray, b: np.ndarray, tol: int = 0) -> np.ndarray:
    """Student's _intersect_with_tol."""
    a = np.unique(a); b = np.unique(b)
    if tol <= 0:
        return np.intersect1d(a, b)
    bset = set(b.tolist())
    hit = []
    for x in a:
        ok = False
        for t in range(-tol, tol + 1):
            if (x + t) in bset:
                ok = True
                break
        if ok:
            hit.append(x)
    return np.array(sorted(set(hit)), dtype=int)


# -----------------------------------------------------------------------------
# Sample generation + training postprocess (student GitHub, verbatim)
# -----------------------------------------------------------------------------
def generate_samples(matrix: np.ndarray, boundary_bins: set, output_path: Path,
                     patch_size: int = 15) -> int:
    """Generate per-bin patches.

    patch_size=15: AxialTAD/ACM convention. pad((7,7),(7,7)); patch = padded[b-7:b+8].
    patch_size=10: deepTAD convention.       pad((4,5),(4,5)); patch = padded[(b+5)-5:(b+5)+5].
    """
    n = len(matrix)
    if patch_size == 15:
        padded = np.pad(matrix, ((7, 7), (7, 7)), mode='constant')
        samples = []
        for bin_idx in range(7, 7 + n):
            patch = padded[bin_idx - 7: bin_idx + 8, bin_idx - 7: bin_idx + 8]
            label = 1 if (bin_idx - 7) in boundary_bins else 0
            samples.append(np.append(patch.flatten(), label))
    elif patch_size == 10:
        padded = np.pad(matrix, ((4, 5), (4, 5)), mode='constant')
        samples = []
        for bin_idx in range(n):
            m = bin_idx + 5
            patch = padded[m - 5: m + 5, m - 5: m + 5]
            label = 1 if bin_idx in boundary_bins else 0
            samples.append(np.append(patch.flatten(), label))
    else:
        raise ValueError(f'unsupported patch_size {patch_size}')
    arr = np.array(samples).astype('float32')
    np.save(output_path, arr)
    return arr.shape[0]


def training_postprocess(submatrix_data: np.ndarray, output_path: Path,
                         patch_size: int = 15) -> tuple[int, int]:
    """Student's training_postprocess: 90° rotation of positives + neg cap at 4× num_pos.
    Returns (num_pos_after_rot, num_neg_sampled).
    """
    pos_samples = submatrix_data[submatrix_data[:, -1] == 1]
    neg_samples = submatrix_data[submatrix_data[:, -1] == 0]

    rotated = []
    for sample in pos_samples:
        m = sample[:-1].reshape(patch_size, patch_size)
        r = np.rot90(m, k=-1).flatten()
        rotated.append(np.append(r, 1))
    rotated = np.array(rotated)

    num_pos = len(pos_samples) + len(rotated)
    neg_limit = min(len(neg_samples), num_pos * 4)
    sampled_neg = neg_samples[np.random.choice(len(neg_samples),
                                                size=neg_limit, replace=False)]

    if len(rotated) > 0:
        final = np.vstack([pos_samples, rotated, sampled_neg])
    else:
        final = np.vstack([pos_samples, sampled_neg]) if len(pos_samples) > 0 else sampled_neg
    np.random.shuffle(final)
    np.save(output_path, final)
    return num_pos, len(sampled_neg)


# -----------------------------------------------------------------------------
# Per-(cell, chr) worker
# -----------------------------------------------------------------------------
def parse_filename(stem: str) -> tuple[str, str]:
    """Parse 'GM12878_chr10_25kb' → ('GM12878', 'chr10')."""
    m = re.match(r'(.+)_(chr[^_]+)_25kb$', stem)
    if not m:
        raise ValueError(f'cannot parse filename: {stem}')
    return m.group(1), m.group(2)


def process_one(cell: str, chrom: str, score_thresh: float = 2.0,
                skip_existing: bool = True, patch_size: int = 15,
                only_refs: tuple = REFS) -> dict:
    """Generate all (or selected) refs for one (cell, chr). Returns stats dict."""
    base = f'{cell}_{chrom}_25kb'
    matrix_path = MATRIX_DIR / f'{base}.txt'
    tad_path    = ONTAD_DIR  / f'{base}.tad'
    bed_path    = HIC_DIR    / f'{base}_boundaries.bed'

    is_test = (cell, chrom) in TEST_SET
    split = 'test' if is_test else 'train'
    patch_dir = f'{patch_size}x{patch_size}'

    # Check whether all expected outputs already exist (resume support)
    expected = []
    for ref in only_refs:
        out_dir = SAMPLES / patch_dir / ref / split
        out_dir.mkdir(parents=True, exist_ok=True)
        raw = out_dir / f'{base}.npy'
        expected.append((ref, raw, out_dir))
    if skip_existing and all(raw.exists() for _, raw, _ in expected) and \
       (is_test or all((od / f'{base}_processed.npy').exists() for _, _, od in expected)):
        return {'cell': cell, 'chrom': chrom, 'split': split, 'status': 'SKIP'}

    # Load matrix once
    M = np.loadtxt(matrix_path, dtype=np.float32)
    n_bins = M.shape[0]

    # Compute the three reference bin sets (only those requested)
    bin_sets = {}
    if 'unfiltered_ontad' in only_refs or 'intersection' in only_refs:
        unf = get_ontad_bins_unfiltered(tad_path)
    if 'filtered_ontad' in only_refs:
        bin_sets['filtered_ontad'] = get_ontad_bins_filtered(tad_path, score_thresh=score_thresh)
    if 'unfiltered_ontad' in only_refs:
        bin_sets['unfiltered_ontad'] = unf
    if 'intersection' in only_refs:
        hic = load_hic_bins_from_bed(bed_path, chrom=chrom)
        bin_sets['intersection'] = intersect_bins(unf, hic)

    stats = {'cell': cell, 'chrom': chrom, 'split': split, 'n_bins': n_bins, 'status': 'OK'}
    for ref, bins in bin_sets.items():
        out_dir = SAMPLES / patch_dir / ref / split
        raw = out_dir / f'{base}.npy'
        n = generate_samples(M, set(bins.tolist()), raw, patch_size=patch_size)
        n_pos = int(len(bins))
        stats[f'{ref}_n'] = n
        stats[f'{ref}_pos'] = n_pos
        if not is_test:
            proc = out_dir / f'{base}_processed.npy'
            data = np.load(raw)
            num_pos_after_rot, num_neg = training_postprocess(data, proc, patch_size=patch_size)
            stats[f'{ref}_proc_pos'] = num_pos_after_rot
            stats[f'{ref}_proc_neg'] = num_neg
    return stats


# -----------------------------------------------------------------------------
# CLI entry
# -----------------------------------------------------------------------------
def discover_jobs() -> list[tuple[str, str]]:
    """List all (cell, chrom) pairs from existing matrix .txt files."""
    jobs = []
    for p in sorted(MATRIX_DIR.glob('*_25kb.txt')):
        cell, chrom = parse_filename(p.stem)
        jobs.append((cell, chrom))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--root',
        type=Path,
        default=Path(_ENV_ROOT) if _ENV_ROOT else None,
        help="Working root directory containing matrix_kr_25kb/, ontad_output/, "
             "hicexplorer_output_v2/, samples/. Defaults to $AXIALTAD_ROOT.",
    )
    ap.add_argument('--cell', help='Restrict to one cell line (e.g., K562)')
    ap.add_argument('--chrom', help='Restrict to one chromosome (e.g., chr1)')
    ap.add_argument('--parallel', type=int, default=8)
    ap.add_argument('--score-thresh', type=float, default=2.0)
    ap.add_argument('--no-skip', action='store_true', help='Reprocess even if outputs exist')
    ap.add_argument('--patch-size', type=int, default=15, choices=[10, 15])
    ap.add_argument('--refs', default=None,
                    help='Comma-separated subset of refs (e.g., filtered_ontad)')
    ap.add_argument('--matrix-dir', type=Path, default=None,
                    help='Override $ROOT/matrix_kr_25kb')
    ap.add_argument('--ontad-dir', type=Path, default=None,
                    help='Override $ROOT/ontad_output')
    ap.add_argument('--hic-dir', type=Path, default=None,
                    help='Override $ROOT/hicexplorer_output_v2')
    ap.add_argument('--samples-root', type=Path, default=None,
                    help='Override $ROOT/samples')
    args = ap.parse_args()
    if args.root is None:
        ap.error("--root is required (or set AXIALTAD_ROOT environment variable)")
    _set_paths(args.root)
    global MATRIX_DIR, ONTAD_DIR, HIC_DIR, SAMPLES
    if args.matrix_dir:  MATRIX_DIR  = args.matrix_dir
    if args.ontad_dir:   ONTAD_DIR   = args.ontad_dir
    if args.hic_dir:     HIC_DIR     = args.hic_dir
    if args.samples_root: SAMPLES    = args.samples_root

    only_refs = REFS if args.refs is None else tuple(r.strip() for r in args.refs.split(','))
    for r in only_refs:
        if r not in REFS:
            print(f'unknown ref {r!r}; valid: {REFS}', file=sys.stderr)
            return 1

    jobs = discover_jobs()
    if args.cell:
        jobs = [(c, ch) for c, ch in jobs if c == args.cell]
    if args.chrom:
        jobs = [(c, ch) for c, ch in jobs if ch == args.chrom]
    if not jobs:
        print('No jobs match', file=sys.stderr)
        return 1

    print(f'Total jobs: {len(jobs)}  parallel={args.parallel}  patch={args.patch_size}  refs={only_refs}')

    # Sanity-check FLAG conditions
    flagged = []

    skip_existing = not args.no_skip
    score = args.score_thresh
    patch = args.patch_size

    if args.parallel == 1:
        results = [process_one(c, ch, score, skip_existing, patch, only_refs) for c, ch in jobs]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=args.parallel) as pool:
            futures = {pool.submit(process_one, c, ch, score, skip_existing, patch, only_refs): (c, ch)
                       for c, ch in jobs}
            for fut in as_completed(futures):
                c, ch = futures[fut]
                try:
                    s = fut.result()
                    results.append(s)
                    if s['status'] == 'SKIP':
                        print(f'SKIP {c} {ch}')
                        continue
                    line = f"OK   {c:8s} {ch:>5s} ({s['split']})  "
                    for ref in only_refs:
                        ref_short = {'filtered_ontad':'F','unfiltered_ontad':'U','intersection':'I'}[ref]
                        n_pos = s.get(f'{ref}_pos', 0); n = s.get(f'{ref}_n', 0)
                        pct = 100.0 * n_pos / n if n > 0 else 0
                        line += f' {ref_short}={n_pos}/{n}({pct:.1f}%)'
                        if n_pos == 0:
                            flagged.append((c, ch, ref, 'zero_positives'))
                        elif pct > 80:
                            flagged.append((c, ch, ref, f'{pct:.1f}%_pos'))
                    print(line)
                except Exception as e:
                    results.append({'cell': c, 'chrom': ch, 'status': 'FAIL', 'error': str(e)})
                    print(f'FAIL {c} {ch}: {e}', file=sys.stderr)

    # Summary
    print('\n=== summary ===')
    n_ok = sum(1 for s in results if s.get('status') == 'OK')
    n_skip = sum(1 for s in results if s.get('status') == 'SKIP')
    n_fail = sum(1 for s in results if s.get('status') == 'FAIL')
    print(f'OK={n_ok}  SKIP={n_skip}  FAIL={n_fail}  TOTAL={len(results)}')
    if flagged:
        print(f'\n=== AUTO-FLAG ({len(flagged)}) ===')
        for c, ch, r, reason in flagged:
            print(f'  {c} {ch} {r}: {reason}')
    return 0 if n_fail == 0 and not flagged else 1


if __name__ == '__main__':
    sys.exit(main())
