#!/usr/bin/env python3
"""Evaluation script for trained LULC segmentation models.

Usage:
    python scripts/evaluate_model.py --checkpoint outputs/checkpoints/default/last.ckpt --dataset potsdam --data_dir data/processed/potsdam
"""

import argparse
import json

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.transferlearning.datasets.registry import get_dataset
from src.transferlearning.trainers.segmentation_trainer import SegmentationTrainer
from src.transferlearning.datasets.base_dataset import NUM_CLASSES

# ensure modules register themselves
import src.transferlearning.models.backbones.resnet       # noqa: F401
import src.transferlearning.models.backbones.efficientnet  # noqa: F401
import src.transferlearning.models.backbones.vit           # noqa: F401
import src.transferlearning.models.backbones.swint         # noqa: F401
import src.transferlearning.models.backbones.vmamba        # noqa: F401
import src.transferlearning.models.backbones.mambavision   # noqa: F401
import src.transferlearning.models.decoders.unet           # noqa: F401
import src.transferlearning.models.decoders.deeplabv3      # noqa: F401
import src.transferlearning.datasets.potsdam               # noqa: F401
import src.transferlearning.datasets.vaihingen             # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LULC Segmentation Model")

    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g. potsdam, vaihingen)")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset root")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_file", type=str, default=None, help="Path to save JSON results")

    return parser.parse_args()


def main():
    args = parse_args()

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
