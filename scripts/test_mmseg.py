#!/usr/bin/env python3
"""Test script for LULC semantic segmentation using MMSegmentation.

Usage:
    # Segmentation metrics only
    python scripts/test_mmseg.py <config> --checkpoint <ckpt>

    # Full report: metrics + FLOPs/params/throughput
    python scripts/test_mmseg.py <config> --checkpoint <ckpt> --full-report

    # Visualize prediction on a full Potsdam image (6000x6000)
    python scripts/test_mmseg.py <config> --checkpoint <ckpt> \
        --visualize data/potsdam/images/top_potsdam_2_10_RGB.tif
"""

import argparse
import ast
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.runner import Runner
from .train_mmseg import build_run_name
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ISPRS_PALETTE = np.array([
    [255, 255, 255],
    [0, 0, 255],
    [0, 255, 255],
    [0, 255, 0],
    [255, 255, 0],
    [255, 0, 0],
], dtype=np.uint8)

IMAGENET_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Test LULC Segmentation Model (MMSegmentation)")
    parser.add_argument("config", type=str, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file (default: <work_dir>/best.ckpt)")
    parser.add_argument("--work-dir", type=str, default=None, help="Working directory for outputs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--full-report", action="store_true", help="Print full report: metrics + inference stats")
    parser.add_argument("--visualize", type=str, default=None, metavar="IMAGE_PATH",
                        help="Path to original Potsdam image (6000x6000) for prediction visualization")
    parser.add_argument("--cfg-options", nargs="+", action="append", default=[], help="Override config options")
    args = parser.parse_args()

    cfg_options = {}
    if args.cfg_options:
        for opt_list in args.cfg_options:
            for opt in opt_list:
                key, value = opt.split("=", 1)
                try:
                    cfg_options[key] = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    cfg_options[key] = value

    if args.batch_size is not None:
        cfg_options["test_dataloader.batch_size"] = args.batch_size

    return args, cfg_options


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_inference(model, dataloader, device, num_warmup=5, num_iters=50):
    model.eval()
    times = []

    with torch.no_grad():
        for i, data_batch in enumerate(dataloader):
            data_batch = model.data_preprocessor(data_batch, False)
            if i < num_warmup:
                _ = model(**data_batch, mode='predict')
                torch.cuda.synchronize()
                continue

            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            _ = model(**data_batch, mode='predict')
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

            if i >= num_warmup + num_iters:
                break

    if not times:
        return {}

    bs = dataloader.batch_size
    avg_time = np.mean(times)
    std_time = np.std(times)
    throughput = bs / avg_time
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return {
        "avg_latency_ms": round(avg_time * 1000, 2),
        "std_latency_ms": round(std_time * 1000, 2),
        "throughput_imgs_per_sec": round(throughput, 1),
        "peak_gpu_memory_mb": round(peak_mem, 1),
        "input_resolution": "256x256",
        "batch_size": bs,
    }


def measure_flops(model, config):
    try:
        from fvcore.nn import FlopCountAnalysis
        device = next(model.parameters()).device
        dummy_input = torch.randn(1, 3, 256, 256).to(device)
        fca = FlopCountAnalysis(model, dummy_input)
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        total_flops = fca.total()
        return total_flops
    except Exception:
        return None


def format_flops(flops):
    if flops is None:
        return "N/A"
    if flops >= 1e12:
        return f"{flops / 1e12:.2f} TFLOPs"
    elif flops >= 1e9:
        return f"{flops / 1e9:.2f} GFLOPs"
    elif flops >= 1e6:
        return f"{flops / 1e6:.2f} MFLOPs"
    return f"{flops:.0f} FLOPs"


def visualize_prediction(runner, image_path, output_dir):
    """Generate prediction visualization for a full Potsdam image (6000x6000).

    Uses sliding window inference with logit averaging (patch=256, stride=128).
    Pads to 6144x6144 (24*256) for even tiling, then crops back.

    Args:
        runner: mmseg Runner with loaded model.
        image_path: Path to original Potsdam image, e.g.
            ``data/potsdam/images/top_potsdam_2_10_RGB.tif``
        output_dir: Output folder. Will contain:
            - image.png: original full-size image
            - label.png: original full-size label (ISPRS RGB colors)
            - pred.png: model prediction (ISPRS RGB colors)

    Label path is derived from image_path by replacing
    ``images/`` with ``labels/`` and ``_RGB.tif`` with ``_label.tif``.
    """
    from PIL import Image

    os.makedirs(output_dir, exist_ok=True)

    model = runner.model
    if hasattr(model, 'module'):
        model = model.module
    model.eval()
    device = next(model.parameters()).device

    label_path = image_path.replace("/images/", "/labels/").replace("_RGB.tif", "_label.tif")

    print(f"  Image:  {image_path}")
    print(f"  Label:  {label_path}")

    img = np.array(Image.open(image_path).convert('RGB'))
    label_rgb = np.array(Image.open(label_path).convert('RGB'))
    orig_h, orig_w = img.shape[:2]

    Image.fromarray(img).save(os.path.join(output_dir, "image.png"))
    Image.fromarray(label_rgb).save(os.path.join(output_dir, "label.png"))
    print(f"  Saved image.png ({img.shape})") # [H, W, C]
    print(f"  Saved label.png ({label_rgb.shape})")

    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) # [1, C, H, W]
    img_tensor = (img_tensor - IMAGENET_MEAN) / IMAGENET_STD
    pad_size = int(np.ceil(orig_h / 256)) * 256
    _, _, H, W = img_tensor.shape
    img_padded = F.pad(img_tensor, (0, pad_size - W, 0, pad_size - H), mode='reflect')

    patch_size = 256
    stride = 128
    num_classes = 6
    batch_size = 8

    positions = [
        (h, w)
        for w in range(0, pad_size - patch_size + 1, stride)
        for h in range(0, pad_size - patch_size + 1, stride)
    ]
    print(f"  Patches: {len(positions)} ({patch_size}x{patch_size}, stride={stride}), padded to {pad_size}x{pad_size}")

    pred_full = torch.zeros(1, num_classes, pad_size, pad_size)
    count_map = torch.zeros(1, 1, pad_size, pad_size)

    with torch.no_grad():
        for i in range(0, len(positions), batch_size):
            batch_positions = positions[i:i + batch_size]
            patches = torch.cat([
                img_padded[:, :, w:w + patch_size, h:h + patch_size]
                for w, h in batch_positions
            ], dim=0).to(device)

            x = model.extract_feat(patches)
            batch_img_metas = [dict(
                img_shape=(patch_size, patch_size),
                ori_shape=(patch_size, patch_size),
                pad_shape=(patch_size, patch_size),
                padding_size=[0, 0, 0, 0],
                )] * len(batch_positions)
            logits = model.decode_head.predict(x, batch_img_metas, model.test_cfg)

            for j, (w, h) in enumerate(batch_positions):
                pred_full[:, :, w:w + patch_size, h:h + patch_size] += logits[j].cpu().unsqueeze(0)
                count_map[:, :, w:w + patch_size, h:h + patch_size] += 1

            done = min(i + batch_size, len(positions))
            if done % (batch_size * 50) == 0 or done == len(positions):
                print(f"    {done}/{len(positions)} patches")

    pred_full /= count_map
    pred_full = pred_full[:, :, :orig_h, :orig_w]
    pred = pred_full.argmax(dim=1).squeeze().numpy()

    pred_color = ISPRS_PALETTE[pred]
    Image.fromarray(pred_color).save(os.path.join(output_dir, "pred.png"))
    print(f"  Saved pred.png ({orig_w}x{orig_h})")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, arr, title in zip(axes, [img, label_rgb, pred_color], ["RGB Image", "Label", "Prediction"]):
        ax.imshow(arr)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "comparison.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved comparison.png")


def main():
    args, cfg_options = parse_args()

    cfg = Config.fromfile(args.config)
    if cfg_options:
        cfg.merge_from_dict(cfg_options)

    if args.work_dir:
        cfg.work_dir = args.work_dir

    run_name = build_run_name(cfg, args.seed)
    for vb in cfg.visualizer.vis_backends:
        if vb.type == "WandbVisBackend":
            vb.init_kwargs.name = f'eval-{run_name}'
            break

    if args.checkpoint is None:
        args.checkpoint = os.path.join(cfg.work_dir, "best.ckpt")
    if not os.path.exists(args.checkpoint):
        best_glob = sorted(glob.glob(os.path.join(cfg.work_dir, "best_mIoU_*.pth")))
        if best_glob:
            args.checkpoint = best_glob[-1]
        else:
            raise FileNotFoundError(f"No checkpoint found at {args.checkpoint} and no best_mIoU_*.pth in {cfg.work_dir}")

    cfg.load_from = args.checkpoint

    cfg.test_evaluator = dict(
        type="IoUMetric",
        iou_metrics=["mIoU", "mDice", "mFscore"],
    )

    _wandb_run_id = None
    if args.visualize:
        for vb in cfg.visualizer.vis_backends:
            if vb.type == "WandbVisBackend":
                if wandb.run is None:
                    wandb.init(**vb.init_kwargs)
                _wandb_run_id = wandb.run.id
                break

    runner = Runner.from_cfg(cfg)

    print("\n" + "=" * 60)
    print("SEGMENTATION METRICS (Test Set)")
    print("=" * 60)
    runner.test()

    if args.full_report:
        print("\n" + "=" * 60)
        print("MODEL & INFERENCE PROFILE")
        print("=" * 60)

        model = runner.model
        device = next(model.parameters()).device

        total_params, trainable_params = count_parameters(model)
        print(f"  Total parameters:     {total_params:,} ({total_params / 1e6:.2f}M)")
        print(f"  Trainable parameters: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")

        flops = measure_flops(model, cfg)
        print(f"  FLOPs (256x256):      {format_flops(flops)}")

        inf_stats = measure_inference(model, runner.test_dataloader, device)
        if inf_stats:
            print(f"  Avg latency:          {inf_stats['avg_latency_ms']} +/- {inf_stats['std_latency_ms']} ms")
            print(f"  Throughput:           {inf_stats['throughput_imgs_per_sec']} imgs/sec")
            print(f"  Peak GPU memory:      {inf_stats['peak_gpu_memory_mb']} MB")

        report = {
            "config": args.config,
            "checkpoint": args.checkpoint,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "flops": flops,
            "inference": inf_stats,
        }

        report_path = os.path.join(runner.work_dir, "test_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n  Report saved to {report_path}")

    if args.visualize:
        print("\n" + "=" * 60)
        print("VISUALIZATION")
        print("=" * 60)
        vis_dir = os.path.join(runner.work_dir, "visualizations")
        visualize_prediction(runner, args.visualize, vis_dir)

        if _wandb_run_id is not None:
            import wandb
            wandb.init(id=_wandb_run_id, resume="allow")
            wandb.log({
                "eval/comparison": wandb.Image(os.path.join(vis_dir, "comparison.png")),
                "eval/image": wandb.Image(os.path.join(vis_dir, "image.png")),
                "eval/label": wandb.Image(os.path.join(vis_dir, "label.png")),
                "eval/pred": wandb.Image(os.path.join(vis_dir, "pred.png")),
            })
            print("  Synced visualizations to wandb")
            wandb.finish()


if __name__ == "__main__":
    main()
