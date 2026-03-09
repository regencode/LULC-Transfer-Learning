import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones.registry import get_backbone
from .decoders.registry import get_decoder


class SegmentationModel(nn.Module):
    """Composes a backbone and decoder into a full segmentation model.

    The backbone produces multi-scale feature maps (stage1..stage4).
    The decoder consumes those features and outputs per-pixel class logits.
    Output is upsampled to match input resolution.
    """

    def __init__(
        self,
        backbone_name: str,
        decoder_name: str,
        num_classes: int,
        pretrained: bool = True,
        backbone_kwargs: dict = None,
        decoder_kwargs: dict = None,
    ):
        super().__init__()
        backbone_kwargs = backbone_kwargs or {}
        decoder_kwargs = decoder_kwargs or {}

        self.backbone = get_backbone(backbone_name, pretrained=pretrained, **backbone_kwargs)
        stage_channels = self.backbone.get_stage_channels()

        self.decoder = get_decoder(
            decoder_name,
            num_classes=num_classes,
            stage_channels=stage_channels,
            **decoder_kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]
        features = self.backbone(x)
        logits = self.decoder(features)
        if logits.shape[2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return logits
