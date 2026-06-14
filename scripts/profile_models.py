#!/usr/bin/env python3
"""Profile model architectures: parameters, FLOPs, inference latency, and GPU memory.

Does not require trained checkpoints — builds models with random weights.
Uses synthetic inputs at the given patch size with batch_size=1 for
standard single-image profiling (latency in ms, FPS = 1/latency).

Usage:
    python scripts/profile_models.py configs/mmseg/potsdam/*.py

    python scripts/profile_models.py \
        configs/mmseg/potsdam/resnet50_deeplabv3plus_patch256_lr1e-4_100e.py \
        configs/mmseg/potsdam/resnet50_upernet_patch256_lr1e-4_100e.py

    python scripts/profile_models.py configs/mmseg/potsdam/*.py \
        --patch-size 512 --num-iters 100 --output results.csv
"""

import argparse
import csv
import gc
import os
import sys
import time

import numpy as np
import torch
from fvcore.nn import FlopCountAnalysis
from mmengine.config import Config
from mmengine.runner import Runner

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile model architectures (params, FLOPs, throughput, memory)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("configs", nargs="+", type=str, help="Config file paths")
    parser.add_argument("--patch-size", type=int, default=512,
                        help="Patch size for synthetic input (default: 512)")
    parser.add_argument("--num-warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--num-iters", type=int, default=100, help="Number of timed inference iterations")
    parser.add_argument("--output", type=str, default="profile_results.csv", help="Output CSV path")
    return parser.parse_args()


def parse_model_info(cfg):
    bb = cfg.model.backbone
    backbone_name = BACKBONE_NAMES.get(bb.type, lambda b: b.type)(bb)
    head_name = HEAD_NAMES.get(cfg.model.decode_head.type, cfg.model.decode_head.type)
    has_aux = cfg.model.auxiliary_head is not None
    return backbone_name, head_name, has_aux


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_flops(model, patch_size=512):
    try:
        device = next(model.parameters()).device
        dummy_input = torch.randn(1, 3, patch_size, patch_size).to(device)
        fca = FlopCountAnalysis(model, dummy_input)
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        return fca.total()
    except Exception:
        return None


def measure_inference(model, patch_size=512, num_warmup=10, num_iters=100):
    """Measure single-image inference latency and FPS using synthetic input.

    Creates a random tensor [1, 3, patch_size, patch_size], passes it through
    the model's data preprocessor, then times model.predict at batch_size=1.

    Returns dict with latency (ms), FPS (1/latency), and peak GPU memory.
    """
    model.eval()
    dummy_input = torch.randn(1, 3, patch_size, patch_size)
    data_batch = model.data_preprocessor(
        {"inputs": dummy_input, "data_samples": None}, False
    )
    times = []

    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for i in range(num_warmup + num_iters):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(**data_batch, mode="predict")
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            if i >= num_warmup:
                times.append(elapsed)

    if not times:
        return {}

    avg_time = np.mean(times)
    std_time = np.std(times)
    fps = 1.0 / avg_time
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return {
        "avg_latency_ms": round(avg_time * 1000, 2),
        "std_latency_ms": round(std_time * 1000, 2),
        "fps": round(fps, 1),
        "peak_gpu_memory_mb": round(peak_mem, 1),
        "input_resolution": f"{patch_size}x{patch_size}",
    }


def format_flops(flops):
    if flops is None:
        return "N/A"
    if flops >= 1e12:
        return f"{flops / 1e12:.2f}T"
    elif flops >= 1e9:
        return f"{flops / 1e9:.2f}G"
    elif flops >= 1e6:
        return f"{flops / 1e6:.2f}M"
    return f"{flops:.0f}"


def format_params(params):
    if params >= 1e6:
        return f"{params / 1e6:.2f}M"
    elif params >= 1e3:
        return f"{params / 1e3:.2f}K"
    return str(params)


def profile_config(config_path, patch_size, num_warmup, num_iters):
    cfg = Config.fromfile(config_path)

    cfg.load_from = None
    cfg.resume = False

    if hasattr(cfg.model, 'backbone') and hasattr(cfg.model.backbone, 'init_cfg'):
        cfg.model.backbone.init_cfg = None
    if hasattr(cfg.model, 'backbone') and hasattr(cfg.model.backbone, 'pretrained'):
        cfg.model.backbone.pretrained = None

    cfg.work_dir = "/tmp/profile_models_workdir"

    for vb in cfg.visualizer.vis_backends:
        if vb.type == "WandbVisBackend":
            vb.init_kwargs = dict(project="lulc-segmentation", config=dict(mode="profile"))
            break

    backbone_name, head_name, has_aux = parse_model_info(cfg)
    model_label = f"{backbone_name}-{head_name}-aux{has_aux}"

    print(f"  Profiling {model_label} ...", end=" ", flush=True)

    runner = Runner.from_cfg(cfg)
    model = runner.model
    if hasattr(model, 'module'):
        model = model.module
    model.eval()

    total_params, trainable_params = count_parameters(model)
    flops = measure_flops(model, patch_size=patch_size)

    inf_stats = measure_inference(model, patch_size=patch_size,
                                  num_warmup=num_warmup, num_iters=num_iters)

    peak_mem = inf_stats.get("peak_gpu_memory_mb", 0) if inf_stats else 0

    print(
        f"params={format_params(total_params)}  "
        f"flops={format_flops(flops)}  "
        f"latency={inf_stats.get('avg_latency_ms', 'N/A')}ms  "
        f"mem={peak_mem:.0f}MB"
    )

    result = {
        "config": os.path.basename(config_path),
        "backbone": backbone_name,
        "head": head_name,
        "aux": has_aux,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "flops": flops,
        "flops_human": format_flops(flops),
        "patch_size": patch_size,
        "avg_latency_ms": inf_stats.get("avg_latency_ms"),
        "std_latency_ms": inf_stats.get("std_latency_ms"),
        "fps": inf_stats.get("fps"),
        "peak_gpu_memory_mb": inf_stats.get("peak_gpu_memory_mb"),
    }

    del runner
    gc.collect()
    torch.cuda.empty_cache()

    return result


def print_table(results):
    hdr = (
        f"{'Backbone':<20} {'Head':<15} {'Aux':<5} {'Params':<12} {'FLOPs':<12} "
        f"{'Latency(ms)':<16} {'FPS':<10} {'GPU Mem(MB)':<12}"
    )
    sep = "-" * len(hdr)
    print(f"\n{sep}\n{hdr}\n{sep}")
    for r in results:
        p = format_params(r["total_params"])
        f = format_flops(r["flops"])
        lat = f"{r['avg_latency_ms']:.2f} +/- {r['std_latency_ms']:.2f}" if r["avg_latency_ms"] else "N/A"
        fps = f"{r['fps']:.1f}" if r["fps"] else "N/A"
        mem = f"{r['peak_gpu_memory_mb']:.1f}" if r["peak_gpu_memory_mb"] else "N/A"
        print(f"{r['backbone']:<20} {r['head']:<15} {str(r['aux']):<5} {p:<12} {f:<12} {lat:<16} {fps:<10} {mem:<12}")
    print(sep)


def write_csv(results, path):
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {path}")


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print("Warning: No GPU detected. Memory and throughput measurements will be unavailable.", file=sys.stderr)

    if len(args.configs) == 1:
        try:
            result = profile_config(
                args.configs[0],
                patch_size=args.patch_size,
                num_warmup=args.num_warmup, num_iters=args.num_iters,
            )
            tmp_path = args.output + ".tmp"
            file_exists = os.path.exists(tmp_path)
            fieldnames = list(result.keys())
            with open(tmp_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(result)
        except Exception as e:
            print(f"  FAILED: {os.path.basename(args.configs[0])}: {e}", file=sys.stderr)
            sys.exit(1)
        return

    import subprocess

    tmp_csv = args.output + ".tmp"
    if os.path.exists(tmp_csv):
        os.remove(tmp_csv)

    print(f"Profiling {len(args.configs)} model(s) in isolated processes ...\n")

    failed = []
    for i, config_path in enumerate(args.configs):
        print(f"[{i + 1}/{len(args.configs)}] {os.path.basename(config_path)}", end=" ", flush=True)
        proc = subprocess.run(
            [
                sys.executable, __file__,
                config_path,
                "--patch-size", str(args.patch_size),
                "--num-warmup", str(args.num_warmup),
                "--num-iters", str(args.num_iters),
                "--output", args.output,
            ],
            capture_output=True, text=True,
        )
        if proc.returncode == 0:
            print("OK")
        else:
            err_msg = proc.stderr.strip().split("\n")[-1] if proc.stderr.strip() else "unknown error"
            print(f"FAILED ({err_msg})")
            failed.append((config_path, err_msg))

    results = []
    if os.path.exists(tmp_csv):
        with open(tmp_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["total_params"] = int(row["total_params"])
                row["trainable_params"] = int(row["trainable_params"])
                row["flops"] = int(row["flops"]) if row["flops"] != "None" else None
                row["aux"] = row["aux"] == "True"
                for key in ["avg_latency_ms", "std_latency_ms", "fps", "peak_gpu_memory_mb"]:
                    row[key] = float(row[key]) if row[key] else None
                results.append(row)

    results.sort(key=lambda r: (r["backbone"], r["head"], r["aux"]))

    print_table(results)
    write_csv(results, args.output)

    if os.path.exists(tmp_csv):
        os.remove(tmp_csv)

    if failed:
        print(f"\n{len(failed)} config(s) failed:", file=sys.stderr)
        for path, err in failed:
            print(f"  {os.path.basename(path)}: {err}", file=sys.stderr)


if __name__ == "__main__":
    main()
