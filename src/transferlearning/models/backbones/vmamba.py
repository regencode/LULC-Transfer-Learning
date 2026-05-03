"""VMamba backbone wrapper.

VMamba is integrated from the official repository:
https://github.com/MzeroMiko/VMamba

It requires cloning the repo and installing its dependencies (mamba-ssm, etc).
The VSSM model class is imported from the VMamba classification codebase.
"""

from typing import List, Dict
import torch
import torch.nn as nn

from .registry import register_backbone

try:
    from classification.models.vmamba import VSSM
    VMAMBA_AVAILABLE = True
except ImportError:
    VMAMBA_AVAILABLE = False

VMAMBA_CONFIGS = {
    "vmamba_tiny": {
        "depths": [2, 2, 9, 2],
        "dims": 96,
        "stage_channels": [96, 192, 384, 768],
    },
    "vmamba_small": {
        "depths": [2, 2, 27, 2],
        "dims": 96,
        "stage_channels": [96, 192, 384, 768],
    },
    "vmamba_base": {
        "depths": [2, 2, 27, 2],
        "dims": 128,
        "stage_channels": [128, 256, 512, 1024],
    },
}


class VMambaBackbone(nn.Module):
    """Wraps the official VMamba (VSSM) model for multi-scale feature extraction.

    VMamba uses 2D Selective Scan (SS2D) to process visual data with
    linear-time complexity. This wrapper extracts hierarchical features
    from the 4 stages for use with segmentation decoders.
    """

    def __init__(self, variant: str = "vmamba_tiny", pretrained: bool = True, checkpoint_path: str = ""):
        super().__init__()
        if not VMAMBA_AVAILABLE:
            raise ImportError(
                "VMamba is not installed. Clone https://github.com/MzeroMiko/VMamba "
                "and ensure it is on your PYTHONPATH."
            )
        if variant not in VMAMBA_CONFIGS:
            raise ValueError(f"Unknown VMamba variant: {variant}")

        config = VMAMBA_CONFIGS[variant]
        self._stage_channels = config["stage_channels"]

        self.model = VSSM(
            depths=config["depths"],
            dims=config["dims"],
            num_classes=0,
        )

        if pretrained and checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]
            self.model.load_state_dict(state_dict, strict=False)

        self._register_hooks()

    def _register_hooks(self):
        self._features = {}
        for i, layer in enumerate(self.model.layers):
            layer.register_forward_hook(self._make_hook(f"stage{i + 1}"))

    def _make_hook(self, name: str):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                feat = output
            else:
                feat = output[0]
            if feat.dim() == 3:
                B, L, C = feat.shape
                H = W = int(L ** 0.5)
                feat = feat.transpose(1, 2).reshape(B, C, H, W)
            elif feat.dim() == 4 and feat.shape[1] != self._stage_channels[0]:
                feat = feat.permute(0, 3, 1, 2)
            self._features[name] = feat
        return hook

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self._features = {}
        self.model(x)
        return self._features

    def get_stage_channels(self) -> List[int]:
        return self._stage_channels


@register_backbone("vmamba_tiny")
def vmamba_tiny_backbone(pretrained: bool = True, checkpoint_path: str = "") -> VMambaBackbone:
    return VMambaBackbone(variant="vmamba_tiny", pretrained=pretrained, checkpoint_path=checkpoint_path)


@register_backbone("vmamba_small")
def vmamba_small_backbone(pretrained: bool = True, checkpoint_path: str = "") -> VMambaBackbone:
    return VMambaBackbone(variant="vmamba_small", pretrained=pretrained, checkpoint_path=checkpoint_path)


@register_backbone("vmamba_base")
def vmamba_base_backbone(pretrained: bool = True, checkpoint_path: str = "") -> VMambaBackbone:
    return VMambaBackbone(variant="vmamba_base", pretrained=pretrained, checkpoint_path=checkpoint_path)
