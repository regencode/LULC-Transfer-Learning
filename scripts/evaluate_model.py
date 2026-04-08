#!/usr/bin/env python3
"""Evaluation script for trained LULC segmentation models.

Usage:
    python scripts/evaluate_model.py --checkpoint outputs/checkpoints/default/last.ckpt --config configs/config.yaml
"""

import argparse
import json
import os
import yaml

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from transferlearning.datasets.registry import get_dataset
from transferlearning.trainers.segmentation_trainer import SegmentationTrainer

# ensure modules register themselves
import transferlearning.models.backbones.resnet       # noqa: F401
import transferlearning.models.backbones.efficientnet  # noqa: F401
import transferlearning.models.backbones.vit           # noqa: F401
import transferlearning.models.backbones.swint         # noqa: F401
import transferlearning.models.backbones.vmamba        # noqa: F401
import transferlearning.models.backbones.mambavision   # noqa: F401
import transferlearning.models.decoders.unet           # noqa: F401
import transferlearning.models.decoders.deeplabv3      # noqa: F401
import transferlearning.datasets.potsdam               # noqa: F401
import transferlearning.datasets.vaihingen             # noqa: F401


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LULC Segmentation Model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")

    # Dataset (from config, can override inline)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)

    # Evaluation options (script defaults, override inline)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_file", type=str, default=None, help="Path to save JSON results")

    args = parser.parse_args()

    # Load config file
    config = {}
    if args.config and os.path.exists(args.config):
        config = load_config(args.config)

    # Merge: CLI args override config (only if not None)
    merged = {**config}
    for k, v in vars(args).items():
        if v is not None:
            merged[k] = v

    class Config:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)

    return Config(merged)


def main():
    args = parse_args()

    print("\n" + "="*60)
    print("Evaluation Configuration")
    print("="*60)
    for key in sorted(vars(args).keys()):
        print(f"  {key}: {getattr(args, key)}")
    print("="*60 + "\n")

    test_dataset = get_dataset(args.dataset, root=args.data_dir, split="test")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = SegmentationTrainer.load_from_checkpoint(args.checkpoint)

    trainer = pl.Trainer(accelerator="auto", devices=1)
    results = trainer.test(model, test_loader)

    if results:
        print("\nTest Results:")
        for key, value in results[0].items():
            print(f"  {key}: {value:.4f}")

        if args.output_file:
            with open(args.output_file, "w") as f:
                json.dump(results[0], f, indent=2)
            print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
