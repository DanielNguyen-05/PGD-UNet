<div align="center">

# PGD-UNet

### Lightweight Polyp Segmentation via Architecture-Aware Pruning & Knowledge Distillation

*A 4-stage compression pipeline combining Structured Pruning, Learnable Channel Gating,*  
*and Knowledge Distillation for real-time colorectal polyp segmentation.*

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Datasets](https://img.shields.io/badge/Datasets-Kvasir--SEG%20%7C%20CVC--ClinicDB-f59e0b?style=flat-square)

**Authors:** Nguyen Hai Dang (23127165) · Nguyen Tran Quoc Duy (23127181)  
**Course:** Computer Vision  
**Instructors:** Vo Hoai Viet · Pham Minh Hoang · Pham Thanh Tung

</div>

---

## Table of Contents

1. [Overview](#1-overview)
2. [Key Features](#2-key-features)
3. [Repository Structure](#3-repository-structure)
4. [Environment Setup](#4-environment-setup)
5. [Dataset Layout](#5-dataset-layout)
6. [Data Preparation](#6-data-preparation)
7. [Training](#7-training)
8. [Evaluation & Testing](#8-evaluation--testing)
9. [Compression Pipeline](#9-compression-pipeline)
10. [CLI Reference](#10-cli-reference)
11. [Reproducibility](#11-reproducibility)
12. [Citation](#12-citation)
13. [License](#13-license)

---

## 1. Overview

This repository provides a full **2D segmentation workspace** targeting **CVC-ClinicDB** and **Kvasir-SEG**, covering data split management, model training, evaluation, and visualization.

On top of the standard segmentation baseline, the project introduces **PGD-UNet** — a four-stage compression pipeline to produce a **lightweight, accurate, and real-time** polyp segmentation model:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         PGD-UNet Compression Pipeline                          │
├────────────┬──────────────────────────┬────────────────────────────────────────┤
│  Stage 1   │  Train Teacher           │  Full-capacity UNet-ResNet152          │
│  Stage 2   │  Structured Pruning      │  Auto-generate architecture blueprint  │
│  Stage 3   │  Build Student           │  Apply Learnable Channel Gating        │
│  Stage 4   │  Distillation Training   │  L_seg + λ₁·L_distill + λ₂·L_sparsity │
└────────────┴──────────────────────────┴────────────────────────────────────────┘
```

### 🎯 Targets

| Metric | Goal |
|--------|------|
| Dice score | ≈ 0.88 – 0.91 |
| Parameters | ≤ 5M |
| Inference speed | ≥ 30 FPS |

---

## 2. Key Features

- **Unified dataloader** that auto-matches image/mask pairs from common folder layouts
- **Multiple segmentation backbones** via a single model factory:

  | Model key | Architecture |
  |-----------|-------------|
  | `unet` | Standard UNet encoder-decoder |
  | `unet_resnet152` | UNet with pretrained ResNet-152 encoder |
  | `resunet` | Residual UNet |
  | `vnet` | Volumetric-style 2D VNet |
  | `unetr` | Transformer-based UNet |
  | `gated_unet` | ⭐ Student model with Learnable Channel Gating *(new)* |

- **Training** with combined CrossEntropy + Dice loss
- **Composite distillation loss:** `L_seg + λ₁·L_distill + λ₂·L_sparsity`
- **Validation and testing** with Dice and HD95 metrics
- **Built-in qualitative outputs:** image / ground-truth / prediction / overlay panel
- **Dataset utilities** for deterministic split generation and statistics reports
- **Structured Pruning** that automatically generates an architecture blueprint
- **Google Colab-compatible** training pipeline

---

## 3. Repository Structure

```text
PGD-UNet/
├── train2d.py                    ← Teacher / baseline training
├── train_compression.py          ← Student training (compression pipeline)
├── test2d.py                     ← Evaluation & visualization
├── requirements.txt
│
├── networks/
│   ├── net_factory.py            ← Model registry
│   ├── common.py
│   ├── unet.py
│   ├── Unet_restnet.py           ← UNet-ResNet152
│   ├── residual_unet.py
│   ├── VNet.py
│   ├── unetr.py
│   └── gated_unet.py             ← ⭐ Student with Learnable Channel Gating
│
├── utils/
│   ├── pruning.py                ← Structured Pruning (Stage 2)
│   ├── compression_loss.py       ← 3-component distillation loss
│   ├── evaluation.py
│   ├── losses.py
│   ├── val_2d.py
│   └── visualization.py
│
├── dataloaders/
│   └── dataset.py
│
├── analysis_data/
│   ├── generate_splits.py
│   ├── analyze_datasets.py
│   └── reports/
│
├── data/
│   ├── CVC-ClinicDB/
│   └── Kvasir-SEG/
│
├── logs/                             ← Checkpoints, metrics, visualizations
└── README.md
```

---

## 4. Environment Setup

> **Requirements:** Python 3.10+

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows PowerShell

# Install dependencies
pip install -r requirements.txt
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('Driver version from torch:', torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'No GPU')"
```

> **Note:** If you need a specific CUDA build, install `torch` and `torchvision` first from the
> [official PyTorch index](https://pytorch.org/get-started/locally/), then install the remaining packages.

---

## 5. Dataset Layout

Place datasets under `data/` with the following structure:

```text
data/
├── CVC-ClinicDB/
│   ├── images/       (accepted aliases: image, original)
│   └── masks/        (accepted aliases: mask, labels, ground_truth)
└── Kvasir-SEG/
    ├── images/
    └── masks/
```

The dataloader **automatically resolves** common folder name variants.

For reproducible experiments, use split manifest files:

```text
data/<dataset>/splits/
├── train.txt
├── val.txt
└── test.txt
```

---

## 6. Data Preparation

### Generate Stable Splits

```bash
python analysis_data/generate_splits.py --dataset all --seed 1337
```

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset` | `all` | `all`, `cvc`, or `kvasir` |
| `--cvc_val_ratio` | `0.1` | Validation fraction for CVC |
| `--cvc_test_ratio` | `0.2` | Test fraction for CVC |
| `--kvasir_val_ratio` | `0.1` | Validation fraction for Kvasir |

### Analyze Dataset Statistics

```bash
python analysis_data/analyze_datasets.py --dataset all
```

Outputs JSON and Markdown summary reports to `analysis_data/reports/`.

---

## 7. Training

### Stage 1 — Train Teacher

**Kvasir-SEG with standard UNet:**

```bash
python train2d.py \
  --dataset kvasir \
  --root_path data/Kvasir-SEG \
  --model unet \
  --train_split train \
  --val_split val \
  --batch_size 8 \
  --max_epochs 100 \
  --eval_interval_epochs 1 \
  --exp teacher_unet
  --deterministic 1 \
  --gpu 0
```

**CVC-ClinicDB with UNet-ResNet152 (pretrained encoder):**

```bash
python train2d.py \
  --dataset cvc \
  --root_path data/CVC-ClinicDB \
  --model unet_resnet152 \
  --train_split train \
  --val_split val \
  --batch_size 8 \
  --max_epochs 100 \
  --eval_interval_epochs 1 \
  --exp teacher_resnet152
  --deterministic 1 \
  --gpu 0
```

**Legacy iteration-based training (optional):**

```bash
python train2d.py \
  --dataset kvasir \
  --root_path data/Kvasir-SEG \
  --model unet \
  --max_iterations 30000 \
  --eval_interval 20 \
  --exp teacher_iter
```

**Training outputs** are saved under `logs/model/supervised/<exp>/`:

```text
logs/model/supervised/<exp>/
├── run_config.json          ← Full configuration snapshot
├── best_checkpoint.json     ← Best checkpoint metadata
├── log.txt                  ← Training log
├── weights/                 ← Saved model weights
└── evaluations/
    └── <dataset>/<model>/<checkpoint>/
```

> After training, the best checkpoint is **automatically evaluated** on the configured final splits (default: `train`, `val`, `test`).

---

### Stage 4 — Train Student (Compression Pipeline)

> ⚠️ Stages 2 and 3 (Structured Pruning and Student construction) are executed **automatically** inside this script. No manual steps required between stages.

```bash
python train_compression.py \
  --dataset kvasir \
  --root_path data/Kvasir-SEG \
  --teacher_model unet \
  --teacher_exp teacher_unet \
  --prune_ratio 0.5 \
  --lambda_distill 0.3 \
  --lambda_sparsity 0.3 \
  --max_epochs 120 \
  --batch_size 12 \
  --exp student_from_unet \
  --gpu 0
```

| Argument | Description |
|----------|-------------|
| `--teacher_model` | Architecture name of the Teacher |
| `--teacher_exp` | Experiment name used when training the Teacher |
| `--prune_ratio` | Fraction of channels to prune (0.0 – 1.0) |
| `--lambda_distill` | Weight for the distillation loss term `λ₁` |
| `--lambda_sparsity` | Weight for the sparsity loss term `λ₂` |

---

## 8. Evaluation & Testing

```bash
python test2d.py \
  --dataset kvasir \
  --root_path data/Kvasir-SEG \
  --model gated_unet \
  --split all \
  --exp student_from_unet
```

| Option | Description |
|--------|-------------|
| `--split` | `train`, `val`, `test`, or `all` |
| `--checkpoint_path` | Path to a specific `.pth` weight file (optional) |

**Evaluation outputs** are grouped by dataset, architecture, checkpoint, and split:

```text
logs/model/supervised/<exp>/evaluations/<dataset>/<model>/<checkpoint>/<split>/
├── case_metrics.csv           ← Per-case Dice and HD95 scores
├── summary.json
├── metrics_summary.json
├── summary.md
├── image/                     ← Input images
├── gt/                        ← Ground-truth masks
├── pred/                      ← Predicted masks
└── panel/                     ← Side-by-side overlay panels
```

A cross-split overview is also written to:

```text
logs/model/supervised/<exp>/evaluations/<dataset>/<model>/<checkpoint>/
└── evaluation_overview.json
```

---

## 9. Compression Pipeline

All four stages map to specific code components as follows:

```
Stage 1 ──► train2d.py
              │  Trains UNet-ResNet152 (Teacher)
              ▼
Stage 2 ──► utils/pruning.py
              │  Structured Pruning → architecture blueprint
              ▼
Stage 3 ──► networks/gated_unet.py
              │  Student model with Learnable Channel Gating
              ▼
Stage 4 ──► train_compression.py
              │  Loss: L_seg + λ₁·L_distill + λ₂·L_sparsity
              ▼
           Lightweight Student (≤ 5M params · ≥ 30 FPS)
```

> Stages 2–4 are handled **automatically** by `train_compression.py`.

---

## 10. CLI Reference

### Shared arguments (`train2d.py` and `test2d.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--root_path` | — | Dataset root directory |
| `--dataset` | — | `cvc`, `cvc_clinicdb`, `kvasir`, `kvasir_seg`, `cyst2d`, `generic` |
| `--model` | — | Model name as registered in `networks/net_factory.py` |
| `--patch_size H W` | `256 256` | Input resolution |
| `--num_classes` | `2` | Number of output classes |
| `--in_channels` | `3` | `1` for grayscale, `3` for RGB |
| `--batch_size` | `8` | Training batch size |
| `--gpu` | `0` | CUDA device ID |
| `--exp` | — | Experiment name (used for output directory) |
| `--deterministic` | `1` | Enable deterministic training |
| `--seed` | `1337` | Random seed |

### Training arguments (`train2d.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--max_epochs` | — | Number of epochs (epoch mode) |
| `--eval_interval_epochs` | `1` | Validate every N epochs |
| `--max_iterations` | — | Legacy stopping criterion (iteration mode) |
| `--eval_interval` | `20` | Validate every N iterations (iteration mode) |
| `--encoder_pretrained` | `0` | Use pretrained encoder weights (`1` to enable) |
| `--final_eval_splits` | `train val test` | Splits evaluated after training |

### Compression arguments (`train_compression.py`)

| Argument | Description |
|----------|-------------|
| `--teacher_model` | Teacher architecture name |
| `--teacher_exp` | Teacher experiment name |
| `--prune_ratio` | Channel pruning ratio (0.0 – 1.0) |
| `--lambda_distill` | Distillation loss weight `λ₁` |
| `--lambda_sparsity` | Sparsity loss weight `λ₂` |

### Test arguments (`test2d.py`)

| Argument | Description |
|----------|-------------|
| `--split` | `train`, `val`, `test`, or `all` |
| `--checkpoint_path` | Optional path to a specific `.pth` weight file |

---

## 11. Reproducibility

- Enable deterministic mode with `--deterministic 1` and fix `--seed`
- Keep split manifests constant when comparing models
- Use `--exp` to isolate checkpoints and results for each experiment
- All run configurations are saved in `run_config.json` for exact replication

---

## 12. Citation

If you use this code in your research or coursework, please cite:

```bibtex
@misc{pgdunet2026,
  author = {Nguyen Hai Dang and Nguyen Tran Quoc Duy},
  title  = {PGD-UNet: Architecture-Aware Pruning and Knowledge Distillation
            with Learnable Channel Gating for Lightweight Real-Time Polyp Segmentation},
  year   = {2026},
  url    = {https://github.com/DanielNguyen-05/PGD-UNet}
}
```

---

<div align="center">
<sub>Built with ❤️ for Computer Vision · Ho Chi Minh City University of Science · 2026</sub>
</div>