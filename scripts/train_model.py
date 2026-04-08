#!/usr/bin/env python3
"""Training script for LULC semantic segmentation.

Usage:
    python scripts/train_model.py --lr 1e-3 --batch_size 16
    # Controlled vars from config.yaml, hyperparameters override inline
"""

import argparse
import os
import yaml

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from torch.utils.data import DataLoader
from typing import Any

from transferlearning.datasets.registry import get_dataset
from transferlearning.trainers.segmentation_trainer import SegmentationTrainer

import transferlearning.datasets.potsdam               # noqa: F401
import transferlearning.datasets.vaihingen             # noqa: F401


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Train LULC Segmentation Model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")

    # Controlled variables (from config, can override inline)
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--decoder", type=str, default=None)
    parser.add_argument("--pretrained", type=str, default=None)  # "true"/"false" or None
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)

    # Hyperparameters (script defaults, override inline)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--scheduler", type=str, default="linear")
    parser.add_argument("--max_epochs", type=int, default=100)

    # Logging (script defaults, override inline)
    parser.add_argument("--experiment_name", type=str, default="default")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", action="store_true", default=True)

    args = parser.parse_args()

    # Load config file
    config = {}
    if args.config:
        config_path = args.config if args.config else "configs/config.yaml"
        if os.path.exists(config_path):
            config = load_config(config_path)

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
    args : Any = parse_args() # silence warnings
    
    print("\n" + "="*60)
    print("Training Configuration")
    print("="*60)
    for key in sorted(vars(args).keys()):
        print(f"  {key}: {getattr(args, key)}")
    print("="*60 + "\n")
    
    pl.seed_everything(args.seed)

    pretrained = args.pretrained if isinstance(args.pretrained, bool) else (args.pretrained.lower() == "true" if args.pretrained else True)

    train_dataset = get_dataset(args.dataset, root=args.data_dir, split="train")
    val_dataset = get_dataset(args.dataset, root=args.data_dir, split="val")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = SegmentationTrainer(
        backbone_name=args.backbone,
        decoder_name=args.decoder,
        num_classes=6,
        pretrained=pretrained,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        max_epochs=args.max_epochs,
    )

    loggers = [
        TensorBoardLogger(save_dir=args.output_dir, name="tensorboard", version=args.experiment_name),
        CSVLogger(save_dir=args.output_dir, name="csv_logs", version=args.experiment_name),
    ]

    if args.use_wandb:
        loggers.append(WandbLogger(
            project="lulc-segmentation",
            name=args.experiment_name,
            save_dir=args.output_dir,
        ))

    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(args.output_dir, "checkpoints", args.experiment_name),
            filename="{epoch}-{val_loss:.4f}-{val_iou:.4f}",
            monitor="val_iou",
            mode="max",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(monitor="val_loss", patience=15, mode="min"),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1,
        logger=loggers,
        callbacks=callbacks,
        precision=args.precision,
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
