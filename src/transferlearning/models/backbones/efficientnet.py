from typing import List, Dict
import torch
import torch.nn as nn
from torchvision.models import (
    efficientnet_b0, efficientnet_b4,
    EfficientNet_B0_Weights, EfficientNet_B4_Weights,
)
from torchvision.models.feature_extraction import create_feature_extractor

from .registry import register_backbone


EFFICIENTNET_CONFIGS = {
    "efficientnet_b0": {
        "model_fn": efficientnet_b0,
        "weights": EfficientNet_B0_Weights.DEFAULT,
        "return_nodes": {
            "features.1": "stage1",
            "features.2": "stage2",
            "features.3": "stage3",
            "features.5": "stage4",
        },
        "stage_channels": [16, 24, 40, 112],
    },
    "efficientnet_b4": {
        "model_fn": efficientnet_b4,
        "weights": EfficientNet_B4_Weights.DEFAULT,
        "return_nodes": {
            "features.1": "stage1",
            "features.2": "stage2",
            "features.3": "stage3",
            "features.5": "stage4",
        },
        "stage_channels": [24, 32, 56, 160],
    },
}


class EfficientNetBackbone(nn.Module):
    def __init__(self, variant: str = "efficientnet_b0", pretrained: bool = True):
        super().__init__()
        if variant not in EFFICIENTNET_CONFIGS:
            raise ValueError(f"Unknown EfficientNet variant: {variant}")

        config = EFFICIENTNET_CONFIGS[variant]
        weights = config["weights"] if pretrained else None
        base_model = config["model_fn"](weights=weights)

        self.feature_extractor = create_feature_extractor(base_model, config["return_nodes"])
        self._stage_channels = config["stage_channels"]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.feature_extractor(x)

    def get_stage_channels(self) -> List[int]:
        return self._stage_channels


@register_backbone("efficientnet_b0")
def efficientnet_b0_backbone(pretrained: bool = True) -> EfficientNetBackbone:
    return EfficientNetBackbone(variant="efficientnet_b0", pretrained=pretrained)


@register_backbone("efficientnet_b4")
def efficientnet_b4_backbone(pretrained: bool = True) -> EfficientNetBackbone:
    return EfficientNetBackbone(variant="efficientnet_b4", pretrained=pretrained)
