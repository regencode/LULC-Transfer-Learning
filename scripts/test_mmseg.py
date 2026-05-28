#!/usr/bin/env python3
"""Test script for LULC semantic segmentation using MMSegmentation.

Usage:
    python scripts/test_mmseg.py configs/mmseg/potsdam/resnet50_unet_lr1e-4_256x256_100e.py --checkpoint outputs/resnet50_unet/best_mIoU.pth
"""

import argparse
import ast

from mmengine.config import Config
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description="Test LULC Segmentation Model (MMSegmentation)")
    parser.add_argument("config", type=str, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--work-dir", type=str, default=None, help="Working directory for outputs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
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


def main():
    args, cfg_options = parse_args()

    cfg = Config.fromfile(args.config)
    if cfg_options:
        cfg.merge_from_dict(cfg_options)

    if args.work_dir:
        cfg.work_dir = args.work_dir

    cfg.load_from = args.checkpoint

    runner = Runner.from_cfg(cfg)
    runner.test()


if __name__ == "__main__":
    main()
