#!/usr/bin/env python
"""Extract KR-balanced dense Hi-C matrix from a .hic file at a chosen resolution.

Output: a tab-separated dense N×N matrix where N = ceil(chrom_length / resolution).
NaN values from KR (low-coverage bins) are zero-filled by default.

Usage:
    python extract_kr_matrix.py <hic_path> <chrom> <out_txt> [--resolution 25000]
"""
import argparse
import os
import sys

import numpy as np

try:
    import hicstraw
except ImportError:
    hicstraw = None


def _require_hicstraw() -> None:
    if hicstraw is None:
        raise ImportError(
            "hic-straw is required for this script. "
            "Install with: pip install hic-straw"
        )


def get_chrom_length(hic_path: str, chrom: str) -> int:
    _require_hicstraw()
    f = hicstraw.HiCFile(hic_path)
    for c in f.getChromosomes():
        if c.name == chrom:
            return c.length
    raise ValueError(f"chromosome {chrom!r} not found in {hic_path}")


def extract(hic_path: str, chrom: str, resolution: int, norm: str = 'KR') -> np.ndarray:
    """Returns dense N×N float32 matrix, NaN→0.

    norm: 'KR' (default) or 'NONE' (raw counts) etc. — passed to hicstraw.straw.
    """
    _require_hicstraw()
    n_bp = get_chrom_length(hic_path, chrom)
    n = (n_bp + resolution - 1) // resolution

    records = hicstraw.straw('observed', norm, hic_path, chrom, chrom, 'BP', resolution)

    if len(records) == 0:
        raise RuntimeError(
            f"hicstraw returned 0 records for {hic_path} chrom={chrom} at {resolution} bp norm={norm}. "
            f"For KR this typically means the .hic file lacks a KR normalization vector for this "
            f"(cell line, chromosome, resolution) combination. Refusing to write an "
            f"all-zero matrix silently. Investigate alternative norms (NONE/VC/VC_SQRT) or "
            f"drop this chromosome from the analysis."
        )

    M = np.zeros((n, n), dtype=np.float32)
    nan_count = 0
    for rec in records:
        v = rec.counts
        i = rec.binX // resolution
        j = rec.binY // resolution
        if i >= n or j >= n:
            continue
        if v != v:  # NaN check (faster than np.isnan for scalar)
            nan_count += 1
            continue
        M[i, j] = v
        if i != j:
            M[j, i] = v

    return M, n, nan_count, len(records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('hic_path')
    ap.add_argument('chrom')
    ap.add_argument('out_txt')
    ap.add_argument('--resolution', type=int, default=25000)
    ap.add_argument('--norm', default='KR', choices=['KR', 'NONE', 'VC', 'VC_SQRT', 'SCALE'])
    args = ap.parse_args()

    M, n, nan_count, n_records = extract(args.hic_path, args.chrom, args.resolution, norm=args.norm)

    # nan_to_num is redundant given we already skipped NaNs, but keep as a safety net
    M = np.nan_to_num(M, nan=0.0)

    np.savetxt(args.out_txt, M, delimiter='\t', fmt='%.6g')

    nz = (M != 0).sum()
    nz_offdiag_upper = ((M != 0) & np.triu(np.ones_like(M, dtype=bool), k=1)).sum()
    n_above_zero_total_pairs = n * (n + 1) // 2
    print(f"OK {os.path.basename(args.hic_path)} chrom={args.chrom} bins={n} "
          f"records={n_records} nan_skipped={nan_count} "
          f"nonzero_cells={int(nz)} ({100*nz/(n*n):.1f}%) "
          f"upper_offdiag_nonzero={int(nz_offdiag_upper)} "
          f"-> {args.out_txt}")


if __name__ == '__main__':
    sys.exit(main() or 0)
