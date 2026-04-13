#!/usr/bin/env python3
"""Test script for trained LULC segmentation models.

Usage:
    python scripts/test_model.py --checkpoint outputs/checkpoints/default/last.ckpt --use_wandb
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Any

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

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
    parser = argparse.ArgumentParser(description="Test LULC Segmentation Model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")

    # WandB options
    parser.add_argument("--wandb_id", type=str, default=None, help="Wandb run ID (auto-detect if not provided)")
    parser.add_argument("--use_wandb", action="store_true", default=False, help="Log metrics to wandb")

    # Dataset (from config, can override inline)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)

    # Test options (script defaults, override inline)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_file", type=str, default=None, help="Path to save JSON results")
    parser.add_argument("--num_vis_samples", type=int, default=4, help="Number of samples for visualization")

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


def get_wandb_run_id(checkpoint_path: str) -> Optional[str]:
    """Auto-detect wandb run ID from checkpoint path."""
    if not checkpoint_path:
        return None
    
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        return None
    
    # Try to find wandb_run_id.txt in checkpoint directory
    # checkpoint path: outputs/checkpoints/{experiment_name}/last.ckpt
    # wandb_run_id.txt should be in: outputs/
    ckpt_dir = ckpt_path.parent.parent  # outputs/checkpoints/{experiment_name}
    output_dir = ckpt_dir.parent  # outputs/
    
    run_id_file = output_dir / "wandb_run_id.txt"
    if run_id_file.exists():
        return run_id_file.read_text().strip()
    
    # Fallback: check current directory
    run_id_file = Path("outputs") / "wandb_run_id.txt"
    if run_id_file.exists():
        return run_id_file.read_text().strip()
    
    return None


def log_visualization(model, test_loader, args):
    """Log image | ground_truth | prediction subplot to wandb."""
    model.eval()
    
    num_samples = min(args.num_vis_samples, args.batch_size)
    
    images_list, labels_list, preds_list = [], [], []
    
    for batch in test_loader:
        images, labels = batch
        with torch.no_grad():
            preds = model(images).argmax(dim=1)
        
        images_list.append(images)
        labels_list.append(labels)
        preds_list.append(preds)
        
        if sum(len(x) for x in images_list) >= num_samples:
            break
    
    images = torch.cat(images_list)[:num_samples]
    labels = torch.cat(labels_list)[:num_samples]
    preds = torch.cat(preds_list)[:num_samples]
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Image
        img = images[i].permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Image")
        axes[i, 0].axis("off")
        
        # Ground Truth
        axes[i, 1].imshow(labels[i].cpu().numpy(), cmap="tab10", vmin=0, vmax=5)
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")
        
        # Prediction
        axes[i, 2].imshow(preds[i].cpu().numpy(), cmap="tab10", vmin=0, vmax=5)
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis("off")
    
    plt.tight_layout()
    
    if args.use_wandb and wandb is not None:
        wandb.log({"test_predictions": wandb.Image(fig)})
    
    if args.output_file:
        output_path = Path(args.output_file).parent / f"{Path(args.output_file).stem}_visualization.png"
        fig.savefig(output_path)
        print(f"\nVisualization saved to {output_path}")
    
    plt.close()


def main():
    args : Any = parse_args()
    
    # Auto-detect wandb_id if not provided and wandb enabled
    if args.use_wandb and not args.wandb_id:
        args.wandb_id = get_wandb_run_id(args.checkpoint)

    print("\n" + "="*60)
    print("Test Configuration")
    print("="*60)
    for key in sorted(vars(args).keys()):
        print(f"  {key}: {getattr(args, key)}")
    print("="*60 + "\n")

    # Initialize wandb if enabled
    if args.use_wandb and args.wandb_id:
        if wandb is None:
            print("Warning: wandb not installed. Install with: pip install wandb")
        else:
            wandb.init(
                id=args.wandb_id,
                project="lulc-segmentation",
                resume="must"
            )
            print(f"Resuming wandb run: {args.wandb_id}")

    test_dataset = get_dataset(args.dataset, root=args.data_dir, split="test", seed=args.seed)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = SegmentationTrainer.load_from_checkpoint(args.checkpoint)

    trainer = pl.Trainer(accelerator="auto", devices=1)
    results = trainer.test(model, test_loader)

    if results:
        print("\nTest Results:")
        for key, value in results[0].items():
            print(f"  {key}: {value:.4f}")
        
        # Log metrics to wandb
        if args.use_wandb and wandb is not None:
            for key, value in results[0].items():
                wandb.log({f"test_{key}": value})
        
        # Log visualization
        log_visualization(model, test_loader, args)

        if args.output_file:
            with open(args.output_file, "w") as f:
                json.dump(results[0], f, indent=2)
            print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
