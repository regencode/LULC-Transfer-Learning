#!/usr/bin/env python3
"""Profile full-tile inference for LULC segmentation models.

Runs sliding-window inference on an entire Potsdam tile (6000x6000) at
two patch sizes (256x256 and 512x512) and measures total inference time,
peak GPU memory, number of patches, and effective throughput.

Does not require trained checkpoints — builds models with random weights.
Each model is profiled in an isolated subprocess to avoid mmengine/wandb
config caching issues.

Usage:
    python scripts/profile_full_tile.py \
        --configs \
            configs/mmseg/potsdam/convnext_b_upernet_patch256_lr1e-4_100e.py \
            configs/mmseg/potsdam/vmamba_b_upernet_patch256_lr1e-4_100e.py \
            configs/mmseg/potsdam/mambavision_b_upernet_patch256_lr1e-4_100e.py \
        --image data/potsdam/images/top_potsdam_2_10_RGB.tif \
        --batch-size 8 \
        --output profiling_full_tile.csv
"""

import argparse
import csv
import gc
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.runner import Runner
from PIL import Image

IMAGENET_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)

BACKBONE_NAMES = {
    "ResNetV1c": lambda bb: f"resnet{bb.depth}",
    "ConvNeXtBackbone": lambda bb: bb.variant.replace("convnext_", "convnext"),
    "VMambaBackbone": lambda bb: bb.variant.replace("vmamba_", "vmamba_").replace("small", "s").replace("base", "b"),
    "MambaVisionBackbone": lambda bb: bb.variant,
}

HEAD_NAMES = {
    "DepthwiseSeparableASPPHead": "deeplabv3plus",
    "UPerHead": "upernet",
}

PATCH_CONFIGS = [
    {"patch_size": 256, "stride": 128},
    {"patch_size": 512, "stride": 256},
]


def parse_args():
    parser = argparse.ArgumentParser(description="Profile full-tile inference")
    parser.add_argument("--configs", nargs="+", type=str, required=False,
                        help="Config file paths (one per model)")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to a Potsdam tile image (6000x6000)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for patch inference (default: 8)")
    parser.add_argument("--num-warmup", type=int, default=1,
                        help="Number of warmup full-tile passes (default: 1)")
    parser.add_argument("--num-iters", type=int, default=3,
                        help="Number of timed full-tile passes (default: 3)")
    parser.add_argument("--output", type=str, default="profiling_full_tile.csv",
                        help="Output CSV path")
    # Internal: when running as subprocess, profile a single config
    parser.add_argument("--single-config", type=str, default=None,
                        help=argparse.SUPPRESS)
    return parser.parse_args()


def parse_model_info(cfg):
    bb = cfg.model.backbone
    backbone_name = BACKBONE_NAMES.get(bb.type, lambda b: b.type)(bb)
    head_name = HEAD_NAMES.get(cfg.model.decode_head.type, cfg.model.decode_head.type)
    has_aux = cfg.model.auxiliary_head is not None
    return backbone_name, head_name, has_aux


def build_model(config_path):
    cfg = Config.fromfile(config_path)
    cfg.load_from = None
    cfg.resume = False

    if hasattr(cfg.model, 'backbone') and hasattr(cfg.model.backbone, 'init_cfg'):
        cfg.model.backbone.init_cfg = None
    if hasattr(cfg.model, 'backbone') and hasattr(cfg.model.backbone, 'pretrained'):
        cfg.model.backbone.pretrained = None

    cfg.work_dir = "/tmp/profile_full_tile_workdir"

    for vb in cfg.visualizer.vis_backends:
        if vb.type == "WandbVisBackend":
            vb.init_kwargs = dict(project="lulc-segmentation", config=dict(mode="profile"))
            break

    backbone_name, head_name, has_aux = parse_model_info(cfg)
    model_label = f"{backbone_name}-{head_name}-aux{has_aux}"

    runner = Runner.from_cfg(cfg)
    model = runner.model
    if hasattr(model, 'module'):
        model = model.module
    model.eval()

    return model, model_label


def load_and_preprocess_image(image_path):
    img = np.array(Image.open(image_path).convert('RGB'))
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
    img_tensor = (img_tensor - IMAGENET_MEAN) / IMAGENET_STD
    return img_tensor, img.shape[:2]


def run_sliding_window(model, img_tensor, orig_h, orig_w,
                       patch_size, stride, batch_size, device):
    _, _, H, W = img_tensor.shape
    pad_size = int(np.ceil(max(H, W) / patch_size)) * patch_size
    img_padded = F.pad(img_tensor, (0, pad_size - W, 0, pad_size - H), mode='reflect')

    num_classes = 6
    positions = [
        (h, w)
        for w in range(0, pad_size - patch_size + 1, stride)
        for h in range(0, pad_size - patch_size + 1, stride)
    ]
    num_patches = len(positions)

    pred_full = torch.zeros(1, num_classes, pad_size, pad_size)
    count_map = torch.zeros(1, 1, pad_size, pad_size)

    with torch.no_grad():
        for i in range(0, num_patches, batch_size):
            batch_positions = positions[i:i + batch_size]
            patches = torch.cat([
                img_padded[:, :, h:h + patch_size, w:w + patch_size]
                for h, w in batch_positions
            ], dim=0).to(device)

            x = model.extract_feat(patches)
            batch_img_metas = [dict(
                img_shape=(patch_size, patch_size),
                ori_shape=(patch_size, patch_size),
                pad_shape=(patch_size, patch_size),
                padding_size=[0, 0, 0, 0],
            )] * len(batch_positions)
            logits = model.decode_head.predict(x, batch_img_metas, model.test_cfg)

            for j, (h, w) in enumerate(batch_positions):
                pred_full[:, :, h:h + patch_size, w:w + patch_size] += logits[j].cpu().unsqueeze(0)
                count_map[:, :, h:h + patch_size, w:w + patch_size] += 1

    pred_full /= count_map
    return num_patches


def profile_single_config(config_path, image_path, batch_size, num_warmup, num_iters):
    print(f"\nBuilding model from {os.path.basename(config_path)} ...")
    model, model_label = build_model(config_path)
    device = next(model.parameters()).device

    print(f"Loading image from {image_path} ...")
    img_tensor, (orig_h, orig_w) = load_and_preprocess_image(image_path)

    results = []

    for pcfg in PATCH_CONFIGS:
        patch_size = pcfg["patch_size"]
        stride = pcfg["stride"]

        print(f"\n  {model_label} | {patch_size}x{patch_size} stride={stride}")

        num_patches = None
        for iteration in range(num_warmup + num_iters):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()

            n = run_sliding_window(model, img_tensor, orig_h, orig_w,
                                   patch_size, stride, batch_size, device)

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            if num_patches is None:
                num_patches = n

            if iteration >= num_warmup:
                peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
                mpx = (orig_h * orig_w) / 1e6 / elapsed
                results.append({
                    "backbone": model_label,
                    "patch_size": patch_size,
                    "stride": stride,
                    "num_patches": num_patches,
                    "tile_time_s": round(elapsed, 3),
                    "peak_gpu_mem_mb": round(peak_mem, 1),
                    "megapixels_per_sec": round(mpx, 1),
                    "per_patch_ms": round(elapsed * 1000 / num_patches, 2),
                })
                print(f"    iter {iteration - num_warmup + 1}: "
                      f"time={elapsed:.3f}s  mem={peak_mem:.0f}MB  "
                      f"MPx/s={mpx:.1f}  patches={num_patches}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return results


def run_subprocess(config_path, image_path, batch_size, num_warmup, num_iters, output_path):
    proc = __import__('subprocess')
    tmp_json = output_path + ".tmp.json"

    cmd = [
        sys.executable, __file__,
        "--single-config", config_path,
        "--image", image_path,
        "--batch-size", str(batch_size),
        "--num-warmup", str(num_warmup),
        "--num-iters", str(num_iters),
        "--output", tmp_json,
    ]

    print(f"  Profiling {os.path.basename(config_path)} ...", end=" ", flush=True)
    result = proc.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        err = result.stderr.strip().split("\n")[-1] if result.stderr.strip() else "unknown"
        print(f"FAILED ({err})")
        return []

    print("OK")

    with open(tmp_json) as f:
        rows = json.load(f)
    os.remove(tmp_json)
    return rows


def print_table(all_results):
    hdr = (
        f"{'Backbone':<30} {'PatchSize':>10} {'Stride':>7} {'Patches':>9} "
        f"{'TileTime(s)':>12} {'PerPatch(ms)':>13} {'PeakMem(MB)':>12} {'MPx/s':>8}"
    )
    sep = "-" * len(hdr)
    print(f"\n{sep}\n{hdr}\n{sep}")
    for r in all_results:
        print(f"{r['backbone']:<30} {r['patch_size']:>10} {r['stride']:>7} {r['num_patches']:>9} "
              f"{r['tile_time_s']:>12.3f} {r['per_patch_ms']:>13.2f} {r['peak_gpu_mem_mb']:>12.0f} "
              f"{r['megapixels_per_sec']:>8.1f}")
    print(sep)


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print("Error: GPU required for full-tile profiling.", file=sys.stderr)
        sys.exit(1)

    # Subprocess mode: profile a single config, write results as JSON
    if args.single_config:
        results = profile_single_config(
            args.single_config, args.image,
            batch_size=args.batch_size,
            num_warmup=args.num_warmup,
            num_iters=args.num_iters,
        )
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        return

    if not args.configs:
        parser.error("--configs is required when not using --single-config")

    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    all_results = []

    print(f"Profiling {len(args.configs)} model(s) in isolated processes ...\n")

    for config_path in args.configs:
        rows = run_subprocess(
            config_path, args.image,
            batch_size=args.batch_size,
            num_warmup=args.num_warmup,
            num_iters=args.num_iters,
            output_path=args.output,
        )
        all_results.extend(rows)

    print_table(all_results)

    fieldnames = ["backbone", "patch_size", "stride", "num_patches",
                  "tile_time_s", "per_patch_ms", "peak_gpu_mem_mb", "megapixels_per_sec"]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
