# Pre-trained Checkpoint

`axialtad_main.weights.h5` is the primary AxialTAD model used in the manuscript.

## Model details

| Property | Value |
|----------|-------|
| Architecture | ACM (Axial Contrast Module), depth D=2 |
| Patch size | 15 × 15 |
| Resolution | 25 kb |
| Reference TADs | Intersection of OnTAD and HiCExplorer hicFindTADs |
| Training optimizer | Adam, learning rate 3e-4 |
| Loss | BinaryCrossentropy (label_smoothing=0.01) |
| Epochs | 30, batch size 256 |
| Random seed | 7 (median of 5 seeds: {0, 7, 42, 100, 123}) |

## Training data

4DN Hi-C contact matrices from four cell lines (GM12878, HeLa-S3, IMR-90,
K562) at 25 kb resolution. Training set: autosomal chromosomes chr1–chr15.
Held-out test: chr20, chr21, chr22 across all four cell lines.

## Performance

See the manuscript for the results. This checkpoint 
corresponds to the median-performing random seed across five training runs.


## Usage

This checkpoint is bundled in the repository — no separate download required.
See the main README "Inference" section for example commands.
