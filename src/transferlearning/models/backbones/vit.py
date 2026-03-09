from typing import List, Dict
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, vit_l_16, ViT_B_16_Weights, ViT_L_16_Weights

from .registry import register_backbone


VIT_CONFIGS = {
    "vit_b_16": {
        "model_fn": vit_b_16,
        "weights": ViT_B_16_Weights.DEFAULT,
        "embed_dim": 768,
        "patch_size": 16,
    },
    "vit_l_16": {
        "model_fn": vit_l_16,
        "weights": ViT_L_16_Weights.DEFAULT,
        "embed_dim": 1024,
        "patch_size": 16,
    },
}


class ViTBackbone(nn.Module):
    """Wraps torchvision ViT to output spatial feature maps for segmentation.

    ViT produces a flat sequence of patch tokens. This wrapper reshapes
    them back into a 2D spatial feature map of shape
    (B, embed_dim, H/patch_size, W/patch_size) so decoders can consume them.

    get_stage_channels returns a single-element list because ViT does not
    have a natural multi-scale hierarchy like CNNs.
    """

    def __init__(self, variant: str = "vit_b_16", pretrained: bool = True):
        super().__init__()
        if variant not in VIT_CONFIGS:
            raise ValueError(f"Unknown ViT variant: {variant}")

        config = VIT_CONFIGS[variant]
        weights = config["weights"] if pretrained else None
        model = config["model_fn"](weights=weights)

        self.patch_size = config["patch_size"]
        self.embed_dim = config["embed_dim"]

        self.patch_embed = model.conv_proj
        self.encoder = model.encoder
        self.ln = nn.LayerNorm(self.embed_dim)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, _, H, W = x.shape
        grid_h, grid_w = H // self.patch_size, W // self.patch_size

        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.encoder(x)
        x = self.ln(x)

        spatial = x.transpose(1, 2).reshape(B, self.embed_dim, grid_h, grid_w)
        return {"stage4": spatial}

    def get_stage_channels(self) -> List[int]:
        return [self.embed_dim]


@register_backbone("vit_b_16")
def vit_b_16_backbone(pretrained: bool = True) -> ViTBackbone:
    return ViTBackbone(variant="vit_b_16", pretrained=pretrained)


@register_backbone("vit_l_16")
def vit_l_16_backbone(pretrained: bool = True) -> ViTBackbone:
    return ViTBackbone(variant="vit_l_16", pretrained=pretrained)
