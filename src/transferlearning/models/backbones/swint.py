from typing import List, Dict
import torch
import torch.nn as nn
from torchvision.models import swin_t, swin_s, swin_b, Swin_T_Weights, Swin_S_Weights, Swin_B_Weights
from torchvision.models.feature_extraction import create_feature_extractor

from .registry import register_backbone


SWINT_CONFIGS = {
    "swin_t": {
        "model_fn": swin_t,
        "weights": Swin_T_Weights.DEFAULT,
        "return_nodes": {
            "features.1": "stage1",
            "features.3": "stage2",
            "features.5": "stage3",
            "features.7": "stage4",
        },
        "stage_channels": [96, 192, 384, 768],
    },
    "swin_s": {
        "model_fn": swin_s,
        "weights": Swin_S_Weights.DEFAULT,
        "return_nodes": {
            "features.1": "stage1",
            "features.3": "stage2",
            "features.5": "stage3",
            "features.7": "stage4",
        },
        "stage_channels": [96, 192, 384, 768],
    },
    "swin_b": {
        "model_fn": swin_b,
        "weights": Swin_B_Weights.DEFAULT,
        "return_nodes": {
            "features.1": "stage1",
            "features.3": "stage2",
            "features.5": "stage3",
            "features.7": "stage4",
        },
        "stage_channels": [128, 256, 512, 1024],
    },
}


class SwinTransformerBackbone(nn.Module):
    """Wraps torchvision Swin Transformer for multi-scale feature extraction.

    Swin Transformer naturally produces hierarchical features at 4 stages,
    but torchvision's output is (B, H*W, C) per stage. This wrapper permutes
    and reshapes outputs to (B, C, H, W) spatial feature maps.
    """

    def __init__(self, variant: str = "swin_t", pretrained: bool = True):
        super().__init__()
        if variant not in SWINT_CONFIGS:
            raise ValueError(f"Unknown Swin variant: {variant}")

        config = SWINT_CONFIGS[variant]
        weights = config["weights"] if pretrained else None
        base_model = config["model_fn"](weights=weights)

        self.feature_extractor = create_feature_extractor(base_model, config["return_nodes"])
        self._stage_channels = config["stage_channels"]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        raw = self.feature_extractor(x)
        out = {}
        for key, feat in raw.items():
            if feat.dim() == 3:
                B, HW, C = feat.shape
                H = W = int(HW ** 0.5)
                feat = feat.transpose(1, 2).reshape(B, C, H, W)
            out[key] = feat
        return out

    def get_stage_channels(self) -> List[int]:
        return self._stage_channels


@register_backbone("swin_t")
def swin_t_backbone(pretrained: bool = True) -> SwinTransformerBackbone:
    return SwinTransformerBackbone(variant="swin_t", pretrained=pretrained)


@register_backbone("swin_s")
def swin_s_backbone(pretrained: bool = True) -> SwinTransformerBackbone:
    return SwinTransformerBackbone(variant="swin_s", pretrained=pretrained)


@register_backbone("swin_b")
def swin_b_backbone(pretrained: bool = True) -> SwinTransformerBackbone:
    return SwinTransformerBackbone(variant="swin_b", pretrained=pretrained)
