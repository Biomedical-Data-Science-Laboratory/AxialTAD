# AxialTAD

> An Axial Contrast Inductive Bias for Parameter-Efficient TAD Boundary Prediction from Hi-C

AxialTAD is a deep learning method that predicts topologically associating
domain (TAD) boundaries from Hi-C contact matrices. The model is built around
the Axial Contrast Module (ACM), a parameter-efficient architectural component
that captures axial (row/column) context contrast in Hi-C patches.

<!-- Status badges (License / Python / TensorFlow) — add after the GitHub repo is public. -->

## Table of Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Data preparation](#data-preparation)
- [Training](#training)
- [Inference](#inference)
- [Post-processing](#post-processing)
- [Pre-trained checkpoint](#pre-trained-checkpoint)
- [Repository layout](#repository-layout)
- [Citation](#citation)
- [License](#license)
- [Authors and Acknowledgements](#authors-and-acknowledgements)
- [Notes for users](#notes-for-users)

## Installation

AxialTAD requires Python 3.11 and TensorFlow 2.19. A GPU is recommended for
training (tested on NVIDIA A6000); inference runs on CPU but slowly.

```bash
git clone https://github.com/Biomedical-Data-Science-Laboratory/AxialTAD.git
cd AxialTAD

conda create -n axialtad python=3.11
conda activate axialtad
pip install -r requirements.txt
```

## Quick start

1. Set the working directory env var:

   ```bash
   export AXIALTAD_ROOT=/path/to/your/workdir
   ```

2. The pre-trained checkpoint is bundled at `checkpoints/axialtad_main.weights.h5`
   (see [Pre-trained checkpoint](#pre-trained-checkpoint) for model details).

3. Run inference on a sample Hi-C patch:

   ```bash
   python scripts/predict.py \
       --arch ACM --patch 15 --D 2 \
       --weights checkpoints/axialtad_main.weights.h5 \
       --input /path/to/your_25kb.npy \
       --output predictions.json
   ```

## Data preparation

AxialTAD expects input as `.npy` patch arrays at **25 kb resolution**, shape
`(N, patch, patch, 1)` where `patch` is 10 or 15. Two preparation steps:

### 1. KR-balanced contact matrix extraction

Given a `.hic` file, extract per-chromosome KR-balanced matrices using
[`hic-straw`](https://github.com/aidenlab/straw):

```bash
python scripts/extract_kr_matrix.py \
    /path/to/sample.hic chr1 /path/to/output/chr1_25kb.txt \
    --resolution 25000 --norm KR
```

For batch extraction across chromosomes, see `examples/extract_kr_bulk.sh`.

### 2. Reference TAD boundaries (for training only)

For training, AxialTAD uses external TAD callers as reference boundaries.
Any caller may be used; this work tested:

- [OnTAD](https://github.com/anlin00007/OnTAD) — compiled from source.
- [HiCExplorer](https://hicexplorer.readthedocs.io) `hicFindTADs` —
  `pip install hicexplorer==3.7.6` (in a separate env if you want to avoid
  TensorFlow conflicts).

After running a TAD caller on each KR matrix, point `generate_samples.py` to
the resulting boundary files:

```bash
python scripts/generate_samples.py --patch-size 15 --parallel 8
```

Inference (`predict.py`) does **not** require reference TADs.

### Input data sources

Hi-C contact data used in this work were obtained from public sources:

- 4D Nucleome Data Portal: https://data.4dnucleome.org
  - GM12878, HeLa-S3, IMR-90, K562

Users may apply AxialTAD to any cell line / sample. The pre-trained checkpoint
generalises beyond the training cell lines (see paper for cross-cell evaluation).

## Training

```bash
python scripts/train.py \
    --arch ACM --patch 15 --D 2 \
    --ref intersection \
    --samples-root $AXIALTAD_ROOT/samples \
    --output $AXIALTAD_ROOT/trained_models/ACM_D2 \
    --gpu 0
```

Key arguments:

- `--arch ACM` — the Axial Contrast Module architecture.
- `--patch {10,15}` — patch size.
- `--D {1,2}` — ACM depth.
- `--ref` — reference TAD source (`filtered_ontad`, `unfiltered_ontad`, `intersection`).

For ablations (vertical-only / horizontal-only ACM, gate-removed), see
`python scripts/train.py --help`. Hyperparameters used in the paper are in
`configs/default.json`.

## Inference

For inference on new Hi-C patches with a pre-trained checkpoint:

```bash
python scripts/predict.py \
    --arch ACM --patch 15 --D 2 \
    --weights checkpoints/axialtad_main.weights.h5 \
    --input /path/to/patches_dir \
    --output predictions.json
```

`--input` accepts either a single `.npy` file or a directory containing
`*_25kb.npy` files. Output is a JSON file with per-sample boundary
probabilities and 0.5-thresholded binary calls.

Optional: `--has-labels` if your `.npy` files have a trailing label column,
to emit precision / recall / F1 metrics alongside predictions.

## Post-processing

The raw probabilities from `predict.py` can be filtered using a Wilcoxon
rank-sum post-hoc test against random patches:

```bash
python scripts/wilcoxon_filter.py --root $AXIALTAD_ROOT --model <model_dir>
```

This produces a filtered boundary call set with statistical-significance
controls.

## Pre-trained checkpoint

The primary AxialTAD model `axialtad_main.weights.h5` is included directly in
this repository under `checkpoints/`. No separate download is required after
cloning.

| File | Size | Description |
|------|------|-------------|
| `checkpoints/axialtad_main.weights.h5` | ~1.7 MB | ACM (D=2), 15×15 patches, 25 kb. Trained on 4DN Hi-C (GM12878, HeLa-S3, IMR-90, K562). |

See [`checkpoints/README.md`](checkpoints/README.md) for full training details
and performance metrics.

## Repository layout

```
AxialTAD/
├── axialtad/                  # Core package
│   ├── __init__.py
│   ├── models.py              # ACM architecture + arch registry
│   └── preprocess.py          # Hi-C patch preprocessing
├── scripts/                   # CLI entry points
│   ├── train.py               # Training driver
│   ├── predict.py             # Inference driver
│   ├── generate_samples.py    # Build training patches from raw + refs
│   ├── extract_kr_matrix.py   # .hic → KR-balanced matrix
│   └── wilcoxon_filter.py     # Wilcoxon post-hoc filter
├── configs/
│   └── default.json           # Paper hyperparameters
├── examples/
│   └── extract_kr_bulk.sh     # Batch KR extraction demo
├── checkpoints/
│   └── README.md              # Pre-trained checkpoint instructions
├── requirements.txt
├── LICENSE
├── CITATION.cff
└── README.md
```

## Citation

If you use AxialTAD in your research, please cite:

```bibtex
@article{choi2026axialtad,
  title   = {AxialTAD: An Axial Contrast Inductive Bias for Parameter-Efficient TAD Boundary Prediction from Hi-C},
  author  = {Choi, Woo-Young and Rhee, Je-Keun},
  year    = {2026},
}
```

See [`CITATION.cff`](CITATION.cff) for machine-readable citation metadata.

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE) for
the full text.

## Authors and Acknowledgements

**Authors**

- Woo-Young Choi — Department of Bioinformatics & Life Science, Soongsil University
- Je-Keun Rhee (corresponding author) — Department of Bioinformatics & Life Science, Soongsil University; AI-Bio Convergence Research Institute, Soongsil University

Hi-C contact data used in this work were obtained from the 4D Nucleome
Project (4DN).

## Notes for users

- AxialTAD currently supports 25 kb resolution.
