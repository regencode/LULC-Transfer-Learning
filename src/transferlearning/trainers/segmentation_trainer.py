from typing import Any, Optional
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import LinearLR
from torchmetrics import Accuracy, Precision, Recall, F1Score, JaccardIndex

from ..datasets.base_dataset import POTSDAM_CLASS_WEIGHTS, VAIHINGEN_CLASS_WEIGHTS
from ..models.segmentation_model import SegmentationModel


class SegmentationTrainer(pl.LightningModule):
    """PyTorch Lightning module for semantic segmentation training.

    Wraps SegmentationModel with training/val/test loops, metrics logging
    (CSV + TensorBoard), and optimizer configuration.
    """

    def __init__(
        self,
        backbone_name: str,
        decoder_name: str,
        num_classes: int,
        dataset: str, # "potsdam" or "vaihingen"
        pretrained: bool = True,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        optimizer: str = "adamw",
        max_epochs: int = 100,
        backbone_kwargs: dict = {},
        decoder_kwargs: dict = {},
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = SegmentationModel(
            backbone_name=backbone_name,
            decoder_name=decoder_name,
            num_classes=num_classes,
            pretrained=pretrained,
            backbone_kwargs=backbone_kwargs,
            decoder_kwargs=decoder_kwargs,
        )

        self.weight = POTSDAM_CLASS_WEIGHTS if dataset == "potsdam" else VAIHINGEN_CLASS_WEIGHTS 
        self.criterion = torch.nn.CrossEntropyLoss(
            weight=self.weight
        )
        print(f"using loss weighting for dataset {dataset}: {self.weight}")

        metric_kwargs = dict(task="multiclass", num_classes=num_classes, average="macro")
        self.train_metrics = self._create_metrics(metric_kwargs)
        self.val_metrics = self._create_metrics(metric_kwargs)
        self.test_metrics = self._create_metrics(metric_kwargs)

    @staticmethod
    def _create_metrics(kwargs: dict) -> torch.nn.ModuleDict:
        return torch.nn.ModuleDict({
            "oa": Accuracy(task=kwargs["task"], num_classes=kwargs["num_classes"]),
            "precision": Precision(**kwargs),
            "recall": Recall(**kwargs),
            "f1": F1Score(**kwargs),
            "iou": JaccardIndex(**kwargs),
        })

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch, metrics: torch.nn.ModuleDict, prefix: str):
        images, masks = batch
        logits = self(images)

        if logits.shape[2:] != masks.shape[1:]:
            logits = F.interpolate(logits, size=masks.shape[1:], mode="bilinear", align_corners=False)

        loss = self.criterion(logits, masks)
        preds = logits.argmax(dim=1)

        for name, metric in metrics.items():
            metric(preds, masks)
            self.log(f"{prefix}_{name}", metric, on_step=False, on_epoch=True, prog_bar=(name in ("iou", "oa")))

        self.log(f"{prefix}_loss", loss, on_step=(prefix == "train"), on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, self.train_metrics, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, self.val_metrics, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, self.test_metrics, "test")

    def configure_optimizers(self):
        if self.hparams.optimizer == "adamw":
            optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == "sgd":
            optimizer = SGD(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {self.hparams.optimizer}")

        scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=self.hparams.max_epochs
        )
        return [optimizer], [scheduler]
