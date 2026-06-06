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

        timm_name = CONVNEXT_VARIANTS[variant]["timm_name"]
        self._out_channels = CONVNEXT_VARIANTS[variant]["channels"]
        self.out_indices = (0, 1, 2, 3)

        self.model = timm.create_model(
            timm_name,
            pretrained=False,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )

        if pretrained:
            self._load_pretrained(pretrained)

    def _load_pretrained(self, path):
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"Pretrained weights not found: {path}. "
                f"Download with: "
                f"python -c \"import timm; timm.create_model('{timm_name}', pretrained=True)\""
            )
        state_dict = torch.load(path, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        self.model.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        features = self.model(x)
        return tuple(features[i] for i in range(4))

    @property
    def out_channels(self):
        return self._out_channels
