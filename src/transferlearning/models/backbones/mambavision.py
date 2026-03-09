"""MambaVision backbone wrapper.

Uses the official NVIDIA MambaVision pip package (mambavision>=1.2.0)
and HuggingFace integration.

Install: pip install mambavision
Repo: https://github.com/NVlabs/MambaVision
"""

from typing import List, Dict
import torch
import torch.nn as nn

from .registry import register_backbone

try:
    from transformers import AutoModel
    MAMBAVISION_HF_AVAILABLE = True
except ImportError:
    MAMBAVISION_HF_AVAILABLE = False

try:
    from mambavision import create_model as mambavision_create_model
    MAMBAVISION_PIP_AVAILABLE = True
except ImportError:
    MAMBAVISION_PIP_AVAILABLE = False


MAMBAVISION_CONFIGS = {
    "mambavision_t": {
        "hf_name": "nvidia/MambaVision-T-1K",
        "pip_name": "mamba_vision_T",
        "stage_channels": [80, 160, 320, 640],
    },
    "mambavision_t2": {
        "hf_name": "nvidia/MambaVision-T2-1K",
        "pip_name": "mamba_vision_T2",
        "stage_channels": [80, 160, 320, 640],
    },
    "mambavision_s": {
        "hf_name": "nvidia/MambaVision-S-1K",
        "pip_name": "mamba_vision_S",
        "stage_channels": [96, 192, 384, 768],
    },
    "mambavision_b": {
        "hf_name": "nvidia/MambaVision-B-1K",
        "pip_name": "mamba_vision_B",
        "stage_channels": [128, 256, 512, 1024],
    },
}


class MambaVisionBackbone(nn.Module):
    """Wraps NVIDIA MambaVision for multi-scale feature extraction.

    MambaVision is a hybrid Mamba-Transformer architecture that produces
    hierarchical features at 4 stages. This wrapper returns a dict of
    stage1..stage4 spatial feature maps (B, C, H, W).

    Supports loading via HuggingFace (preferred) or the mambavision pip package.
    """

    def __init__(self, variant: str = "mambavision_t", pretrained: bool = True):
        super().__init__()
        if variant not in MAMBAVISION_CONFIGS:
            raise ValueError(f"Unknown MambaVision variant: {variant}")

        config = MAMBAVISION_CONFIGS[variant]
        self._stage_channels = config["stage_channels"]

        if pretrained and MAMBAVISION_HF_AVAILABLE:
            self.model = AutoModel.from_pretrained(config["hf_name"], trust_remote_code=True)
        elif MAMBAVISION_PIP_AVAILABLE:
            self.model = mambavision_create_model(config["pip_name"], pretrained=pretrained)
        else:
            raise ImportError(
                "Neither 'transformers' nor 'mambavision' packages are available. "
                "Install with: pip install mambavision  or  pip install transformers"
            )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        _, features = self.model(x)
        return {f"stage{i + 1}": feat for i, feat in enumerate(features)}

    def get_stage_channels(self) -> List[int]:
        return self._stage_channels


@register_backbone("mambavision_t")
def mambavision_t_backbone(pretrained: bool = True) -> MambaVisionBackbone:
    return MambaVisionBackbone(variant="mambavision_t", pretrained=pretrained)


@register_backbone("mambavision_t2")
def mambavision_t2_backbone(pretrained: bool = True) -> MambaVisionBackbone:
    return MambaVisionBackbone(variant="mambavision_t2", pretrained=pretrained)


@register_backbone("mambavision_s")
def mambavision_s_backbone(pretrained: bool = True) -> MambaVisionBackbone:
    return MambaVisionBackbone(variant="mambavision_s", pretrained=pretrained)


@register_backbone("mambavision_b")
def mambavision_b_backbone(pretrained: bool = True) -> MambaVisionBackbone:
    return MambaVisionBackbone(variant="mambavision_b", pretrained=pretrained)
