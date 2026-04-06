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

from transferlearning.datasets.registry import get_dataset
from transferlearning.trainers.segmentation_trainer import SegmentationTrainer
from transferlearning.datasets.base_dataset import NUM_CLASSES

# ensure backbone/decoder modules are imported so they register themselves
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train LULC Segmentation Model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file") 

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
