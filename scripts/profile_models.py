#!/usr/bin/env python3
"""Profile model architectures: parameters, FLOPs, inference throughput, and GPU memory.

Does not require trained checkpoints — builds models with random weights.

Usage:
    python scripts/profile_models.py configs/mmseg/potsdam/*.py

    python scripts/profile_models.py \\
        configs/mmseg/potsdam/resnet50_deeplabv3plus_patch256_lr1e-4_100e.py \\
        configs/mmseg/potsdam/resnet50_upernet_patch256_lr1e-4_100e.py

    python scripts/profile_models.py configs/mmseg/potsdam/*.py \\
        --batch-size 16 --num-iters 100 --output results.csv
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
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference measurement")
    parser.add_argument("--num-warmup", type=int, default=5, help="Number of warmup iterations")
    parser.add_argument("--num-iters", type=int, default=50, help="Number of timed inference iterations")
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


def measure_flops(model):
    try:
        device = next(model.parameters()).device
        dummy_input = torch.randn(1, 3, 256, 256).to(device)
        fca = FlopCountAnalysis(model, dummy_input)
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        return fca.total()
    except Exception:
        return None


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


def profile_config(config_path, batch_size, num_warmup, num_iters):
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

    if batch_size != cfg.test_dataloader.batch_size:
        cfg.test_dataloader.batch_size = batch_size

    backbone_name, head_name, has_aux = parse_model_info(cfg)
    model_label = f"{backbone_name}-{head_name}-aux{has_aux}"

    print(f"  Profiling {model_label} ...", end=" ", flush=True)

    runner = Runner.from_cfg(cfg)
    model = runner.model
    if hasattr(model, 'module'):
        model = model.module
    model.eval()
    device = next(model.parameters()).device

    total_params, trainable_params = count_parameters(model)
    flops = measure_flops(model)

    torch.cuda.reset_peak_memory_stats()
    inf_stats = measure_inference(model, runner.test_dataloader, device, num_warmup, num_iters)

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
        "avg_latency_ms": inf_stats.get("avg_latency_ms"),
        "std_latency_ms": inf_stats.get("std_latency_ms"),
        "throughput_imgs_per_sec": inf_stats.get("throughput_imgs_per_sec"),
        "peak_gpu_memory_mb": inf_stats.get("peak_gpu_memory_mb"),
    }

    del runner
    gc.collect()
    torch.cuda.empty_cache()

    return result


def print_table(results):
    hdr = f"{'Backbone':<20} {'Head':<15} {'Aux':<5} {'Params':<12} {'FLOPs':<12} {'Latency(ms)':<14} {'Throughput':<14} {'GPU Mem(MB)':<12}"
    sep = "-" * len(hdr)
    print(f"\n{sep}\n{hdr}\n{sep}")
    for r in results:
        p = format_params(r["total_params"])
        f = format_flops(r["flops"])
        lat = f"{r['avg_latency_ms']:.2f} +/- {r['std_latency_ms']:.2f}" if r["avg_latency_ms"] else "N/A"
        tp = f"{r['throughput_imgs_per_sec']:.1f}" if r["throughput_imgs_per_sec"] else "N/A"
        mem = f"{r['peak_gpu_memory_mb']:.1f}" if r["peak_gpu_memory_mb"] else "N/A"
        print(f"{r['backbone']:<20} {r['head']:<15} {str(r['aux']):<5} {p:<12} {f:<12} {lat:<14} {tp:<14} {mem:<12}")
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

    results = []
    failed = []

    print(f"Profiling {len(args.configs)} model(s) ...\n")

    for config_path in args.configs:
        try:
            result = profile_config(config_path, args.batch_size, args.num_warmup, args.num_iters)
            results.append(result)
        except Exception as e:
            print(f"  FAILED: {os.path.basename(config_path)}: {e}", file=sys.stderr)
            failed.append((config_path, str(e)))

    results.sort(key=lambda r: (r["backbone"], r["head"], r["aux"]))

    print_table(results)
    write_csv(results, args.output)

    if failed:
        print(f"\n{len(failed)} config(s) failed:", file=sys.stderr)
        for path, err in failed:
            print(f"  {os.path.basename(path)}: {err}", file=sys.stderr)


if __name__ == "__main__":
    main()
