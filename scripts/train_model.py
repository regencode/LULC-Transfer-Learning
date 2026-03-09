#!/usr/bin/env python3
"""Training script for LULC semantic segmentation.

Usage:
    python scripts/train_model.py --backbone resnet50 --decoder unet --dataset potsdam --data_dir data/processed/potsdam
"""

import argparse
import os

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from torch.utils.data import DataLoader

from src.transferlearning.datasets.registry import get_dataset
from src.transferlearning.trainers.segmentation_trainer import SegmentationTrainer
from src.transferlearning.datasets.base_dataset import NUM_CLASSES

# ensure backbone/decoder modules are imported so they register themselves
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
    parser = argparse.ArgumentParser(description="Train LULC Segmentation Model")

    parser.add_argument("--backbone", type=str, required=True, help="Backbone name (e.g. resnet50, swin_t, vmamba_tiny)")
    parser.add_argument("--decoder", type=str, required=True, help="Decoder name (e.g. unet, deeplabv3plus)")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g. potsdam, vaihingen)")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset root")

    parser.add_argument("--num_classes", type=int, default=NUM_CLASSES)
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no_pretrained", action="store_true", help="Disable ImageNet pretrained weights")

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "plateau", "none"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--experiment_name", type=str, default="default")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--precision", type=int, default=32, choices=[16, 32])
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()
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
