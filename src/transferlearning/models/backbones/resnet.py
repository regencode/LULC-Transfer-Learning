from typing import List, Dict
import torch
import torch.nn as nn
from torchvision.models import resnet50, resnet101, ResNet50_Weights, ResNet101_Weights
from torchvision.models.feature_extraction import create_feature_extractor

from .registry import register_backbone


RESNET_CONFIGS = {
    "resnet50": {"model_fn": resnet50, "weights": ResNet50_Weights.DEFAULT},
    "resnet101": {"model_fn": resnet101, "weights": ResNet101_Weights.DEFAULT},
}

RETURN_NODES = {
    "layer1": "stage1",
    "layer2": "stage2",
    "layer3": "stage3",
    "layer4": "stage4",
}

FEATURE_CHANNELS = [256, 512, 1024, 2048]


class ResNetBackbone(nn.Module):
    def __init__(self, variant: str = "resnet50", pretrained: bool = True):
        super().__init__()
        if variant not in RESNET_CONFIGS:
            raise ValueError(f"Unknown ResNet variant: {variant}")

        config = RESNET_CONFIGS[variant]
        weights = config["weights"] if pretrained else None
        base_model = config["model_fn"](weights=weights)

        self.feature_extractor = create_feature_extractor(base_model, RETURN_NODES)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.feature_extractor(x)

    @staticmethod
    def get_stage_channels() -> List[int]:
        return FEATURE_CHANNELS


@register_backbone("resnet50")
def resnet50_backbone(pretrained: bool = True) -> ResNetBackbone:
    return ResNetBackbone(variant="resnet50", pretrained=pretrained)


@register_backbone("resnet101")
def resnet101_backbone(pretrained: bool = True) -> ResNetBackbone:
    return ResNetBackbone(variant="resnet101", pretrained=pretrained)
