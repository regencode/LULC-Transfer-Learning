#!/usr/bin/env python3
"""Training script for LULC semantic segmentation using MMSegmentation.

Usage:
    python scripts/train_mmseg.py configs/mmseg/potsdam/resnet50_unet_lr1e-4_256x256_100e.py
    python scripts/train_mmseg.py configs/mmseg/potsdam/resnet50_unet_lr1e-4_256x256_100e.py --work-dir outputs/resnet50_unet
"""

import argparse
import ast
import glob
import os

from mmengine.config import Config
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description="Train LULC Segmentation Model (MMSegmentation)")
    parser.add_argument("config", type=str, help="Path to config file")
    parser.add_argument("--work-dir", type=str, default=None, help="Working directory to save logs and models")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--amp", action="store_true", default=False, help="Enable automatic mixed precision training")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume from the latest checkpoint")
    parser.add_argument("--cfg-options", nargs="+", action="append", default=[], help="Override config options (key=value)")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size for train/val/test")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override max epochs")
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
        cfg_options["train_dataloader.batch_size"] = args.batch_size
        cfg_options["val_dataloader.batch_size"] = args.batch_size
        cfg_options["test_dataloader.batch_size"] = args.batch_size
    if args.lr is not None:
        cfg_options["optim_wrapper.optimizer.lr"] = args.lr
    if args.max_epochs is not None:
        cfg_options["train_cfg.max_epochs"] = args.max_epochs

    return args, cfg_options


BACKBONE_NAMES = {
    "ResNetV1c": lambda bb: f"resnet{bb.depth}",
    "VMambaBackbone": lambda bb: bb.variant.replace("vmamba_", "vmamba_").replace("small", "s").replace("base", "b"),
    "MambaVisionBackbone": lambda bb: bb.variant,
}

HEAD_NAMES = {
    "DepthwiseSeparableASPPHead": "deeplabv3plus",
    "UPerHead": "upernet",
}

DATASET_NAMES = {
    "ISPRSPotsdamDataset": "potsdam",
}


def build_run_name(cfg, seed):
    bb = cfg.model.backbone
    backbone_name = BACKBONE_NAMES[bb.type](bb)
    head_name = HEAD_NAMES[cfg.model.decode_head.type]
    dataset_name = DATASET_NAMES[cfg.train_dataloader.dataset.type]
    has_aux = cfg.model.auxiliary_head is not None
    return f"{dataset_name}-{backbone_name}{head_name}-seed{seed}-aux{has_aux}"


def main():
    args, cfg_options = parse_args()

    cfg = Config.fromfile(args.config)
    if cfg_options:
        cfg.merge_from_dict(cfg_options)

    if args.work_dir:
        cfg.work_dir = args.work_dir

    cfg.randomness = dict(seed=args.seed)

    run_name = build_run_name(cfg, args.seed)
    for vb in cfg.visualizer.vis_backends:
        if vb.type == "WandbVisBackend":
            vb.init_kwargs.name = run_name
            break

    if args.amp:
        cfg.optim_wrapper.type = "AmpOptimWrapper"
        cfg.optim_wrapper.loss_scale = "dynamic"

    runner = Runner.from_cfg(cfg)

    if args.resume:
        ckpt_dir = cfg.work_dir
        last_ckpt = os.path.join(ckpt_dir, "last_checkpoint")
        if os.path.exists(last_ckpt):
            with open(last_ckpt) as f:
                ckpt_path = f.read().strip()
            if not os.path.isabs(ckpt_path):
                ckpt_path = os.path.join(ckpt_dir, ckpt_path)
        else:
            candidates = sorted(glob.glob(os.path.join(ckpt_dir, "epoch_*.pth")))
            if not candidates:
                candidates = sorted(glob.glob(os.path.join(ckpt_dir, "best_*.pth")))
            if candidates:
                ckpt_path = candidates[-1]
            else:
                print(f"No checkpoints found in {ckpt_dir}, starting from scratch")
                ckpt_path = None
        if ckpt_path:
            runner.resume(ckpt_path)
        else:
            runner.train()
    else:
        runner.train()

    best_ckpts = sorted(glob.glob(os.path.join(cfg.work_dir, "best_mIoU_*.pth")))
    if best_ckpts:
        link_path = os.path.join(cfg.work_dir, "best.ckpt")
        if os.path.islink(link_path) or os.path.exists(link_path):
            os.remove(link_path)
        os.symlink(os.path.basename(best_ckpts[-1]), link_path)
        print(f"Created symlink: {link_path} -> {os.path.basename(best_ckpts[-1])}")


if __name__ == "__main__":
    main()
