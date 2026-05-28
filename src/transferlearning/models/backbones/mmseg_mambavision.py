"""MambaVision backbone wrapper for MMSegmentation.

Wraps the existing MambaVision model to conform to mmseg's backbone interface:
forward() returns a tuple of multi-scale feature tensors.
"""

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmseg.registry import MODELS

try:
    from .MambaVisionModels.mamba_vision_modified import MambaVision
    MAMBAVISION_AVAILABLE = True
except ImportError:
    MAMBAVISION_AVAILABLE = False

MAMBAVISION_CONFIGS = {
    "mambavision_t": dict(
        dims=[80, 160, 320, 640],
        depths=[1, 3, 8, 4],
        num_heads=[2, 4, 8, 16],
        window_size=[8, 8, 14, 7],
        mlp_ratio=4,
        drop_path_rate=0.2,
        url="https://huggingface.co/nvidia/MambaVision-T-1K/resolve/main/mambavision_tiny_1k.pth.tar",
    ),
    "mambavision_t2": dict(
        dims=[80, 160, 320, 640],
        depths=[1, 3, 11, 4],
        num_heads=[2, 4, 8, 16],
        window_size=[8, 8, 14, 7],
        mlp_ratio=4,
        drop_path_rate=0.2,
        url="https://huggingface.co/nvidia/MambaVision-T2-1K/resolve/main/mambavision_tiny2_1k.pth.tar",
    ),
    "mambavision_s": dict(
        dims=[96, 192, 384, 768],
        depths=[3, 3, 7, 5],
        num_heads=[2, 4, 8, 16],
        window_size=[8, 8, 14, 7],
        mlp_ratio=4,
        drop_path_rate=0.2,
        url="https://huggingface.co/nvidia/MambaVision-S-1K/resolve/main/mambavision_small_1k.pth.tar",
    ),
    "mambavision_b": dict(
        dims=[128, 256, 512, 1024],
        depths=[3, 3, 10, 5],
        num_heads=[2, 4, 8, 16],
        window_size=[8, 8, 14, 7],
        mlp_ratio=4,
        drop_path_rate=0.3,
        layer_scale=1e-5,
        url="https://huggingface.co/nvidia/MambaVision-B-1K/resolve/main/mambavision_base_1k.pth.tar",
    ),
    "mambavision_l": dict(
        dims=[196, 392, 784, 1568],
        depths=[3, 3, 10, 5],
        num_heads=[4, 8, 16, 32],
        window_size=[8, 8, 14, 7],
        mlp_ratio=4,
        drop_path_rate=0.3,
        layer_scale=1e-5,
        url="https://huggingface.co/nvidia/MambaVision-L-1K/resolve/main/mambavision_large_1k.pth.tar",
    ),
}


@MODELS.register_module()
class MambaVisionBackbone(BaseModule):
    arch_settings = MAMBAVISION_CONFIGS

    def __init__(
        self,
        variant="mambavision_t",
        pretrained=None,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(init_cfg=init_cfg)
        if not MAMBAVISION_AVAILABLE:
            raise ImportError("MambaVision is not available. Install mamba-ssm.")
        if variant not in self.arch_settings:
            raise ValueError(f"Unknown MambaVision variant: {variant}. "
                             f"Available: {list(self.arch_settings.keys())}")

        cfg = self.arch_settings[variant]
        self.model = MambaVision(
            in_chans=3,
            dims=cfg["dims"],
            depths=cfg["depths"],
            window_size=cfg["window_size"],
            num_heads=cfg["num_heads"],
            mlp_ratio=cfg["mlp_ratio"],
            drop_path_rate=cfg.get("drop_path_rate", 0.2),
            layer_scale=cfg.get("layer_scale", None),
            layer_scale_conv=None,
            **kwargs,
        )

        if pretrained:
            self._load_pretrained(pretrained, cfg.get("url"))

        self.out_indices = (0, 1, 2, 3)
        self._out_channels = cfg["dims"]

    def _load_pretrained(self, path, url=None):
        from pathlib import Path
        if not Path(path).is_file() and url:
            torch.hub.download_url_to_file(url=url, dst=path)
        self.model._load_state_dict(path, strict=False)

    def forward(self, x):
        features_dict = self.model(x)
        return tuple(features_dict[f"stage{i+1}"] for i in range(4))

    @property
    def out_channels(self):
        return self._out_channels
