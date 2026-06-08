"""ConvNeXt backbone wrapper for MMSegmentation.

Wraps timm's ConvNeXt to conform to mmseg's backbone interface:
forward() returns a tuple of multi-scale feature tensors.
"""

import torch
from mmengine.model import BaseModule
from mmseg.registry import MODELS
from pathlib import Path
import timm

CONVNEXT_VARIANTS = {
    "convnext_small": {
        "timm_name": "convnext_small.fb_in1k",
        "channels": [96, 192, 384, 768],
    },
    "convnext_base": {
        "timm_name": "convnext_base.fb_in1k",
        "channels": [128, 256, 512, 1024],
    },
}


@MODELS.register_module()
class ConvNeXtBackbone(BaseModule):

    def __init__(
        self,
        variant="convnext_small",
        pretrained=None,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(init_cfg=init_cfg)
        if variant not in CONVNEXT_VARIANTS:
            raise ValueError(
                f"Unknown ConvNeXt variant: {variant}. "
                f"Available: {list(CONVNEXT_VARIANTS.keys())}"
            )

        self.timm_name = CONVNEXT_VARIANTS[variant]["timm_name"]
        self._out_channels = CONVNEXT_VARIANTS[variant]["channels"]
        self.out_indices = (0, 1, 2, 3)

        self.model = timm.create_model(
            self.timm_name,
            pretrained=False,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )

        if pretrained:
            self._load_pretrained(pretrained)

    @staticmethod
    def _remap_state_dict(state_dict):
        remapped = {}
        for key, value in state_dict.items():
            if key.startswith(("head.", "head_norm.")):
                continue
            new_key = key.replace("stem.0", "stem_0").replace("stem.1", "stem_1")
            for i in range(4):
                new_key = new_key.replace(f"stages.{i}.", f"stages_{i}.")
            remapped[new_key] = value
        return remapped

    def _load_pretrained(self, path):
        if not Path(path).is_file():
            m = timm.create_model(self.timm_name, pretrained=True)
            torch.save(m.state_dict(), path)
            del m
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        state_dict = self._remap_state_dict(state_dict)
        self.model.load_state_dict(state_dict, strict=True)

    def forward(self, x):
        features = self.model(x)
        return tuple(features[i] for i in range(4))

    @property
    def out_channels(self):
        return self._out_channels
