# LULC Transfer Learning

Land-Use/Land-Cover semantic segmentation using transfer learning with MMSegmentation. Studies the effect of backbone architecture and decoder choice on ISPRS Potsdam aerial imagery.

## Setup

```bash
pip install -e .
pip install timm fvcore wandb

# For MambaVision backbones (requires CUDA toolkit / nvcc):
pip install mamba-ssm
```

## End-to-End Pipeline

### Step 1: Prepare Raw Data

Extracts images and labels from `Potsdam.zip`, converts RGB labels to class-index maps (0-5), generates train/val/test splits at the image level (80/10/10).

```bash
python scripts/prepare_potsdam.py \
    --source data/Potsdam.zip \
    --dest data/potsdam
```

**Output:**

```
data/potsdam/
├── images/       # 38 original RGB orthophotos (6000x6000 .tif)
├── labels/       # 38 original RGB label TIFs (ISPRS colors)
├── ann_dir/      # 38 class-index PNGs (0-5, used for training)
└── splits/       # train.txt, val.txt, test.txt (filename stems)
```

| Class | Index | ISPRS Color |
|-------|-------|-------------|
| Impervious surface | 0 | White |
| Building | 1 | Blue |
| Low vegetation | 2 | Cyan |
| Tree | 3 | Green |
| Car | 4 | Yellow |
| Clutter | 5 | Red |

### Step 2: Patchify

Splits full-resolution images into 256x256 patches with stride 128. Generates per-patch split files (patches from the same image stay in the same split to prevent data leakage).

```bash
python scripts/patchify_potsdam.py \
    --src data/potsdam \
    --dst data/potsdam_patch256 \
    --patch-size 256 \
    --stride 128
```

**Output:**

```
data/potsdam_patch256/
├── images/       # ~80,408 image patches (256x256 .tif)
├── ann_dir/      # ~80,408 class-index label patches (256x256 .png)
└── splits/       # train.txt (~63,480), val.txt (~8,464), test.txt (~8,464)
```

Steps 1 and 2 are run once and shared across all experiments.

### Step 3: Train

```bash
python scripts/train_mmseg.py <CONFIG> \
    --work-dir outputs/<EXPERIMENT_NAME> \
    --batch-size 8 \
    --max-epochs 100
```

**Flags:**

| Flag | Description |
|------|-------------|
| `--work-dir` | Output directory for checkpoints, logs |
| `--batch-size` | Override batch size (default 32, use 8 for 8GB GPU) |
| `--max-epochs` | Override max epochs (default 100) |
| `--lr` | Override learning rate (default 1e-4) |
| `--amp` | Enable mixed precision training |
| `--resume` | Resume from last checkpoint in work-dir |
| `--cfg-options` | Override arbitrary config keys |

Training logs to **Weights & Biases** (project: `lulc-segmentation`) and saves:
- `best_mIoU_epoch_XX.pth` — best validation mIoU checkpoint
- `epoch_X.pth` — last 3 epoch checkpoints
- Per-epoch validation: mIoU, mDice, mFscore (F1/Precision/Recall), aAcc
- Early stopping after 10 epochs without mIoU improvement

**To resume after interruption:**

```bash
python scripts/train_mmseg.py <CONFIG> \
    --work-dir outputs/<EXPERIMENT_NAME> \
    --batch-size 8 \
    --max-epochs 100 \
    --resume
```

### Step 4: Evaluate

**Segmentation metrics only (mIoU, F1, Precision, Recall, OA):**

```bash
python scripts/test_mmseg.py <CONFIG> \
    --checkpoint outputs/<EXPERIMENT_NAME>/best_mIoU_epoch_XX.pth
```

**Full report (+ parameters, FLOPs, throughput, GPU memory):**

```bash
python scripts/test_mmseg.py <CONFIG> \
    --checkpoint outputs/<EXPERIMENT_NAME>/best_mIoU_epoch_XX.pth \
    --full-report
```

This outputs:

| Metric | Source |
|--------|--------|
| IoU / mIoU per class | IoUMetric |
| Dice / mDice per class | IoUMetric |
| F1 / Precision / Recall per class | IoUMetric (mFscore) |
| Overall Accuracy (aAcc) | IoUMetric |
| Total / Trainable parameters | `sum(p.numel())` |
| GFLOPs (256x256 input) | `fvcore.FlopCountAnalysis` |
| Avg latency (ms) | Timed inference loop |
| Throughput (imgs/sec) | Timed inference loop |
| Peak GPU memory (MB) | `torch.cuda.max_memory_allocated` |

Saves `test_report.json` to the work directory.

**Visualize prediction on a full 6000x6000 Potsdam image:**

```bash
python scripts/test_mmseg.py <CONFIG> \
    --checkpoint outputs/<EXPERIMENT_NAME>/best_mIoU_epoch_XX.pth \
    --visualize data/potsdam/images/top_potsdam_2_10_RGB.tif
```

Uses sliding window inference with logit averaging (patch=256, stride=128, padded to 6144x6144). Saves to `outputs/<EXPERIMENT_NAME>/visualizations/`:

| File | Description |
|------|-------------|
| `image.png` | Original 6000x6000 orthophoto |
| `label.png` | Ground truth (ISPRS RGB colors) |
| `pred.png` | Model prediction (ISPRS RGB colors) |

**All three modes combined:**

```bash
python scripts/test_mmseg.py <CONFIG> \
    --checkpoint outputs/<EXPERIMENT_NAME>/best_mIoU_epoch_XX.pth \
    --full-report \
    --visualize data/potsdam/images/top_potsdam_2_10_RGB.tif
```

---

## Experiment Configs

### Model Matrix

All combinations of **6 backbones** x **2 decoders** = **12 experiments**.

Every config uses:
- FCNHead deep supervision at backbone level 2 (aux loss weight = 0.4)
- AdamW optimizer, lr=1e-4, linear LR decay over 100 epochs
- Class-weighted cross-entropy: `[0.35, 0.45, 0.50, 0.72, 6.47, 2.33]`
- Early stopping (patience=10, monitor mIoU)

| | DeepLabV3+ | UPerNet |
|---|---|---|
| **ResNet50** | `resnet50_deeplabv3plus_patch256_lr1e-4_100e.py` | `resnet50_upernet_patch256_lr1e-4_100e.py` |
| **ResNet101** | `resnet101_deeplabv3plus_patch256_lr1e-4_100e.py` | `resnet101_upernet_patch256_lr1e-4_100e.py` |
| **VMamba-S** | `vmamba_s_deeplabv3plus_patch256_lr1e-4_100e.py` | `vmamba_s_upernet_patch256_lr1e-4_100e.py` |
| **VMamba-B** | `vmamba_b_deeplabv3plus_patch256_lr1e-4_100e.py` | `vmamba_b_upernet_patch256_lr1e-4_100e.py` |
| **MambaVision-S** | `mambavision_s_deeplabv3plus_patch256_lr1e-4_100e.py` | `mambavision_s_upernet_patch256_lr1e-4_100e.py` |
| **MambaVision-B** | `mambavision_b_deeplabv3plus_patch256_lr1e-4_100e.py` | `mambavision_b_upernet_patch256_lr1e-4_100e.py` |

All configs are in `configs/mmseg/potsdam/`.

### Ablation: No Deep Supervision

Configs with `auxiliary_head=None` for studying the effect of deep supervision:

- `resnet101_{deeplabv3plus,upernet}_noaux_patch256_lr1e-4_100e.py`
- `vmamba_b_{deeplabv3plus,upernet}_noaux_patch256_lr1e-4_100e.py`
- `mambavision_b_{deeplabv3plus,upernet}_noaux_patch256_lr1e-4_100e.py`

### Pretrained Weights

| Backbone | Weight Source | Auto-download |
|----------|--------------|---------------|
| ResNet50 | `open-mmlab://resnet50_v1c` | Yes |
| ResNet101 | `open-mmlab://resnet101_v1c` | Yes |
| VMamba-S | `/tmp/vssm1_small_0230s_ckpt_epoch_222.pth` | No (manual) |
| VMamba-B | `/tmp/vssm1_base_0229s_ckpt_epoch_260.pth` | No (manual) |
| MambaVision-S | `/tmp/mambavision_small_1k.pth.tar` | Yes (from HuggingFace) |
| MambaVision-B | `/tmp/mambavision_base_1k.pth.tar` | Yes (from HuggingFace) |

VMamba weights must be downloaded from [HuggingFace](https://huggingface.co/MzeroMiko/VMamba) and placed in `/tmp/`.

---

## Example: Full Run for One Experiment

```bash
# 1. Prepare data (run once)
python scripts/prepare_potsdam.py --source data/Potsdam.zip --dest data/potsdam
python scripts/patchify_potsdam.py --src data/potsdam --dst data/potsdam_patch256 --patch-size 256 --stride 128

# 2. Train
python scripts/train_mmseg.py \
    configs/mmseg/potsdam/resnet50_deeplabv3plus_patch256_lr1e-4_100e.py \
    --work-dir outputs/resnet50_deeplabv3plus \
    --batch-size 8 \
    --max-epochs 100

# 3. Evaluate
python scripts/test_mmseg.py \
    configs/mmseg/potsdam/resnet50_deeplabv3plus_patch256_lr1e-4_100e.py \
    --checkpoint outputs/resnet50_deeplabv3plus/best_mIoU_epoch_XX.pth \
    --full-report \
    --visualize data/potsdam/images/top_potsdam_2_10_RGB.tif
```

---

## Project Structure

```
LULC-Transfer-Learning/
├── configs/mmseg/
│   ├── _base_/
│   │   ├── models/                  # 12 base model configs (backbone + decoder + aux)
│   │   ├── datasets/potsdam_patch256.py
│   │   ├── schedules/schedule_adamw_100e.py
│   │   └── default_runtime.py       # WandB logging, early stopping
│   └── potsdam/                     # 18 experiment configs (12 main + 6 ablation)
├── scripts/
│   ├── prepare_potsdam.py           # Step 1: Extract + convert labels + split
│   ├── patchify_potsdam.py          # Step 2: Create 256x256 patches
│   ├── train_mmseg.py               # Step 3: Train with mmengine Runner
│   └── test_mmseg.py                # Step 4: Evaluate + profile + visualize
├── src/transferlearning/
│   ├── datasets/mmseg_isprs.py      # ISPRSPotsdamDataset (registered with mmseg)
│   ├── models/
│   │   ├── backbones/
│   │   │   ├── mmseg_vmamba.py      # VMambaBackbone wrapper
│   │   │   ├── mmseg_mambavision.py # MambaVisionBackbone wrapper
│   │   │   ├── VMambaModels/        # Core VMamba implementation
│   │   │   └── MambaVisionModels/   # Core MambaVision implementation
│   │   └── heads/unet_head.py       # Custom UNetHead (not used in current matrix)
│   └── utils/losses.py              # FocalLoss, DiceLoss, CombinedLoss
├── data/                            # Created by prepare/patchify scripts
├── outputs/                         # Created during training
└── requirements.txt
```
