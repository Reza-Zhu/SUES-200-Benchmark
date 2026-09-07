# SUES-200 Benchmark

SUES-200 is a multi-height, multi-scene cross-view image matching benchmark
between UAV/drone and satellite images. The repository contains the original
dual-branch baselines together with a maintained data preparation, training,
feature extraction, and retrieval evaluation pipeline.

The benchmark paper was published in IEEE Transactions on Circuits and Systems
for Video Technology. The dataset is available for academic research only.

## Dataset and pretrained weights

Dataset links:

- [Google Drive](https://drive.google.com/file/d/1UyVyFJ_pRaJHIr_eBY2HL7gkS5y9UxqI/view?usp=share_link)
- Baidu Pan: `https://pan.baidu.com/s/1mrd-7ADm57_OchAvO1XmNw` (提取码：`p836`)
- Tianyi: `https://cloud.189.cn/t/yMnaEnR322Yj` (提取码：`veh7`)

Pretrained weights: `https://pan.baidu.com/s/1aq51FLfg3bPG4xoNW1Usxw?pwd=rbnu`
(提取码：`rbnu`).

The raw archive should contain both views:

```text
SUES-200-512x512/
├── drone_view_512/
│   └── 0001/150/*.jpg
└── satellite-view/
    └── 0001/0.png
```

Drone images are organized by scene and height (`150`, `200`, `250`, `300`).
Satellite images are organized by scene and are shared across heights.

## Installation

Install a PyTorch build appropriate for the target CUDA version first, then
install the remaining dependencies:

```bash
python -m pip install torch torchvision
python -m pip install -r requirements.txt
```

`imgaug` is only required by the uncertainty/robustness evaluation entrypoint;
the normal training and retrieval paths do not import it eagerly.

## Configuration

Edit [`settings.yaml`](settings.yaml) for the target machine. Important fields:

| Field | Meaning |
| --- | --- |
| `dataset_path` | Raw/prepared SUES-200 root directory |
| `weight_save_path` | Checkpoint and evaluation output directory |
| `model` | Backbone name, for example `resnet`, `vit`, or `LPN` |
| `height` | Drone height: `150`, `200`, `250`, or `300` |
| `pretrained` | Whether training initializes timm backbones with pretrained weights |
| `batch_size` | Batch size for each view loader |
| `num_workers` / `prefetch_factor` | Image loading throughput controls |
| `eval_chunk_size` | GPU memory/throughput trade-off for batched retrieval |
| `eval_amp` | Optional mixed precision during evaluation; `false` is reproducible default |

Set `pretrained: false` when the server has no cached timm weights or no network
access. The evaluator always loads checkpoints without downloading pretrained
weights.

## End-to-end workflow

### 1. Prepare ImageFolder links

From the repository root:

```bash
python script/split_datasets.py --path /path/to/SUES-200-512x512
```

The script reads [`script/indexs.yaml`](script/indexs.yaml) relative to itself
and creates:

```text
SUES-200-512x512/
├── Training/<height>/{drone,satellite}/<scene>/
└── Testing/<height>/{query_drone,query_satellite,
                      gallery_drone,gallery_satellite}/<scene>/
```

Directory symlinks are used by default, so the raw images are not duplicated.
Use copying only when the filesystem does not support symlinks:

```bash
python script/split_datasets.py \
  --path /path/to/SUES-200-512x512 \
  --mode copy
```

The script is idempotent for links that already point to the correct source.
It stops on missing view/scene/height directories instead of silently creating
an incomplete benchmark.

### 2. Train

Run the configured number of epochs:

```bash
python train.py --cfg settings.yaml
```

Run a one-epoch end-to-end check without changing the YAML file:

```bash
python train.py --cfg settings.yaml --epochs 1
```

Training uses independent drone and satellite batches. If the two views contain
different numbers of images, the shorter loader is restarted so the longer
loader is not silently truncated. The best finite epoch is saved as
`<weight_save_path>/<model>_<height>_<timestamp>/net_<epoch>.pth`, together with
`settings_saved.yaml`.

### 3. Evaluate a checkpoint

For an existing checkpoint directory:

```bash
python test_and_evaluate.py \
  --cfg settings.yaml \
  --name resnet_200_YYYY-MM-DD-HH:MM:SS \
  --seq 1 \
  --dist Cos
```

For one checkpoint file, use the direct form:

```bash
python test_and_evaluate.py \
  --cfg settings.yaml \
  --checkpoint /path/to/net_041.pth \
  --dist Cos
```

The evaluator computes both directions:

- `query_drone → gallery_satellite`
- `query_satellite → gallery_drone`

It reports Recall@1/5/10, Recall@1%, AP, and elapsed time. Features for the
four views are extracted once per checkpoint, then retrieval is performed in
chunks. Results are written to `<weight_save_path>/<name>.csv` and text files
under the checkpoint directory.

Euclidean and Manhattan distance are also available:

```bash
python test_and_evaluate.py --cfg settings.yaml --name <name> --dist Eu
python test_and_evaluate.py --cfg settings.yaml --name <name> --dist Man
```

### 4. Optional evaluations

Robustness evaluation requires `imgaug`:

```bash
python test_and_evaluate_uncertainties.py \
  --cfg settings.yaml \
  --types rain fog snow flip black \
  --heights 150 200 250 300
```

Multi-query pooling for drone queries:

```bash
python multi_test_and_evaluate_pooling.py \
  --cfg settings.yaml \
  --multi 2 \
  --type ave
```

`test.py` remains a compatibility alias for the maintained evaluator.
`evaluate.py` remains available for legacy `pytorch_result.mat` feature dumps.

## Verification

Run the lightweight retrieval tests after installing the dependencies:

```bash
python -m unittest discover -s tests -v
```

For a quick operational check, use `--epochs 1` and a small
`eval_chunk_size`. A one-epoch run confirms data loading, model construction,
forward/backward, AMP/device handling, and checkpoint writing; it does not
represent converged benchmark quality.

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
