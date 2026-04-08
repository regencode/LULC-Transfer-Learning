#!/usr/bin/env python3
"""Training script for LULC semantic segmentation.

Usage:
    python scripts/train_model.py --backbone resnet50 --decoder unet --dataset potsdam --data_dir data/processed/potsdam
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
    parser.add_argument("--config", type=str, help="Path to config file")

    # Model
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--decoder", type=str, default=None)
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--pretrained", action="store_true", default=True)

    # Data
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)

    # Training
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--scheduler", type=str, default="linear")
    parser.add_argument("--max_epochs", type=int, default=100)

    # Logging
    parser.add_argument("--experiment_name", type=str, default="default")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", action="store_true", default=True)

    args = parser.parse_args()

    # Load config file if provided
    config = {}
    if args.config:
        config = load_config(args.config)

    # Merge: CLI args override config
    merged = {**config, **{k: v for k, v in vars(args).items() if v is not None}}

    class Config:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)

    return Config(merged)


def main():
    args : Any = parse_args() # silence warnings
    pl.seed_everything(args.seed)

    pretrained = not args.no_pretrained

    train_dataset = get_dataset(args.dataset, root=args.data_dir, split="train")
    val_dataset = get_dataset(args.dataset, root=args.data_dir, split="val")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = SegmentationTrainer(
        backbone_name=args.backbone,
        decoder_name=args.decoder,
        num_classes=args.num_classes,
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
