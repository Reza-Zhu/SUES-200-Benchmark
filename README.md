<div align="center">

# SUES-200 Benchmark

### Multi-height · Multi-scene · Cross-view Image Matching

UAV / Drone ↔ Satellite

<p>
  <a href="https://arxiv.org/abs/2204.10704">Paper</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#evaluation">Evaluation</a> ·
  <a href="mailto:rzzhu24@m.fudan.edu.cn">Contact</a>
</p>

<p><strong>Author contact:</strong> rzzhu24@m.fudan.edu.cn</p>

</div>

---

<p align="center">
  <img src="assets/sues200-readme-hero.png" alt="SUES-200 cross-view matching between UAV and satellite imagery" width="100%">
</p>

---

## Overview

SUES-200 is a benchmark for matching UAV/drone images with satellite images
across multiple flight heights and scenes. This repository provides the
original dual-branch baselines together with a maintained end-to-end pipeline
for dataset preparation, training, feature extraction, and retrieval
evaluation.

> [!IMPORTANT]
> SUES-200 is available for academic research only. Please follow the dataset
> usage terms when downloading or redistributing data and weights.

### Highlights

| Capability | Description |
| --- | --- |
| **Multi-height data** | Supports 150 m, 200 m, 250 m, and 300 m drone imagery |
| **Efficient preparation** | Creates directory symlinks by default; raw images are not duplicated |
| **Dual-branch models** | ResNet, SE-ResNet, ViT, LPN, VGG, DenseNet, EfficientNet, and related backbones |
| **Bidirectional retrieval** | Evaluates drone → satellite and satellite → drone |
| **Efficient evaluation** | Extracts each view once and performs chunked batched retrieval |
| **Reproducible checks** | Includes one-epoch validation and lightweight retrieval unit tests |

## Contents

- [Dataset and pretrained weights](#dataset-and-pretrained-weights)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Optional evaluations](#optional-evaluations)
- [Verification](#verification)
- [Citation](#citation)

## Dataset and pretrained weights

Dataset downloads:

- [Google Drive](https://drive.google.com/file/d/1UyVyFJ_pRaJHIr_eBY2HL7gkS5y9UxqI/view?usp=share_link)
- Baidu Pan: `https://pan.baidu.com/s/1mrd-7ADm57_OchAvO1XmNw` · 提取码：`p836`
- Tianyi: `https://cloud.189.cn/t/yMnaEnR322Yj` · 提取码：`veh7`

Pretrained weights:

- Baidu Pan: `https://pan.baidu.com/s/1aq51FLfg3bPG4xoNW1Usxw?pwd=rbnu` · 提取码：`rbnu`

The raw archive must contain both views:

```text
SUES-200-512x512/
├── drone_view_512/
│   └── 0001/150/*.jpg
└── satellite-view/
    └── 0001/0.png
```

Drone images are grouped by scene and height. Satellite images are grouped by
scene and shared across heights.

## Quick start

### 1. Install dependencies

Install a PyTorch build matching the target CUDA version, then install the
project dependencies:

```bash
python -m pip install torch torchvision
python -m pip install -r requirements.txt
```

The normal training and retrieval paths do not require `imgaug` at import
time. Install it when running uncertainty/robustness evaluation.

### 2. Prepare the dataset

Run from the repository root:

```bash
python script/split_datasets.py --path /path/to/SUES-200-512x512
```

The script reads [`script/indexs.yaml`](script/indexs.yaml) relative to itself
and creates the ImageFolder layout expected by the training and evaluation
scripts:

```text
SUES-200-512x512/
├── Training/<height>/{drone,satellite}/<scene>/
└── Testing/<height>/{query_drone,query_satellite,
                      gallery_drone,gallery_satellite}/<scene>/
```

Directory symlinks are used by default. To create independent copies instead:

```bash
python script/split_datasets.py \
  --path /path/to/SUES-200-512x512 \
  --mode copy
```

The preparation step is idempotent for correct existing links and stops with a
clear error when a view, scene, or height is missing.

### 3. Train

Train using the configured number of epochs:

```bash
python train.py --cfg settings.yaml
```

Run a fast end-to-end pipeline check without changing the YAML file:

```bash
python train.py --cfg settings.yaml --epochs 1
```

Training uses independent drone and satellite batches. If the two views have
different numbers of images, the shorter loader is restarted instead of
silently truncating the longer loader. The best finite epoch is saved as:

```text
<weight_save_path>/<model>_<height>_<timestamp>/net_<epoch>.pth
```

The effective configuration is saved alongside the checkpoint as
`settings_saved.yaml`.

## Configuration

Edit [`settings.yaml`](settings.yaml) for the target machine.

| Field | Purpose |
| --- | --- |
| `dataset_path` | Raw/prepared SUES-200 root directory |
| `weight_save_path` | Checkpoint and evaluation output directory |
| `model` | Backbone, such as `resnet`, `vit`, or `LPN` |
| `height` | Drone height: `150`, `200`, `250`, or `300` |
| `pretrained` | Initialize training backbones from cached timm weights |
| `batch_size` | Batch size for each view loader |
| `num_workers` | Number of image-loading workers |
| `prefetch_factor` | Batches prefetched by each worker |
| `eval_chunk_size` | Memory/throughput trade-off during retrieval |
| `eval_amp` | Optional mixed precision during evaluation |

Set `pretrained: false` when pretrained timm weights are not cached locally or
the machine has no network access. Evaluation checkpoint loading is offline-safe
and never downloads pretrained weights.

## Evaluation

### Evaluate a checkpoint directory

```bash
python test_and_evaluate.py \
  --cfg settings.yaml \
  --name resnet_200_YYYY-MM-DD-HH:MM:SS \
  --seq 1 \
  --dist Cos
```

### Evaluate one checkpoint file directly

```bash
python test_and_evaluate.py \
  --cfg settings.yaml \
  --checkpoint /path/to/net_041.pth \
  --dist Cos
```

The evaluator reports:

- Recall@1, Recall@5, and Recall@10
- Recall@1%
- Average Precision (AP)
- Elapsed evaluation time

Both retrieval directions are evaluated:

```text
query_drone     → gallery_satellite
query_satellite → gallery_drone
```

Features for the four views are extracted once per checkpoint. Similarity or
distance matrices are then computed in chunks to reduce repeated GPU launches
and control memory usage. Results are saved as CSV and text files below
`weight_save_path`.

Available distance metrics:

```bash
python test_and_evaluate.py --cfg settings.yaml --name <name> --dist Cos
python test_and_evaluate.py --cfg settings.yaml --name <name> --dist Eu
python test_and_evaluate.py --cfg settings.yaml --name <name> --dist Man
```

## Optional evaluations

### Robustness to uncertainty

Requires `imgaug`:

```bash
python test_and_evaluate_uncertainties.py \
  --cfg settings.yaml \
  --types rain fog snow flip black \
  --heights 150 200 250 300
```

### Multi-query pooling

```bash
python multi_test_and_evaluate_pooling.py \
  --cfg settings.yaml \
  --multi 2 \
  --type ave
```

`test.py` is retained as a compatibility alias for the maintained evaluator.
`evaluate.py` remains available for legacy `pytorch_result.mat` feature dumps.

## Verification

Run the retrieval unit tests after installing dependencies:

```bash
python -m unittest discover -s tests -v
```

For a quick operational check, use `--epochs 1`. This verifies data loading,
model construction, forward/backward, device/AMP handling, and checkpoint
writing; it does not represent converged benchmark quality.

## Citation

```text
@ARTICLE{zhu2023sues,
  author={Zhu, Runzhe and Yin, Ling and Yang, Mingze and Wu, Fei and Yang, Yuncheng and Hu, Wenbo},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  title={SUES-200: A Multi-height Multi-scene Cross-view Image Benchmark Across Drone and Satellite},
  year={2023},
  doi={10.1109/TCSVT.2023.3249204}
}
```

<div align="center">

Questions or collaboration: [rzzhu24@m.fudan.edu.cn](mailto:rzzhu24@m.fudan.edu.cn)

</div>
