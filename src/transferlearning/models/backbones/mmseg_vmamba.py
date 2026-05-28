"""VMamba backbone wrapper for MMSegmentation.

Wraps the existing Backbone_VSSM from VMambaModels to conform to mmseg's
backbone interface: forward() returns a tuple of multi-scale feature tensors.
"""

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmseg.registry import MODELS

from .VMambaModels.vmamba import Backbone_VSSM

VMAMBA_CONFIGS = {
    "vmamba_tiny": dict(
        depths=[2, 2, 8, 2],
        dims=96,
        drop_path_rate=0.2,
        ssm_d_state=1,
        ssm_ratio=1.0,
        forward_type="v05_noz",
        mlp_ratio=4.0,
    ),
    "vmamba_small": dict(
        depths=[2, 2, 15, 2],
        dims=96,
        drop_path_rate=0.3,
        ssm_d_state=1,
        ssm_ratio=2.0,
        forward_type="v05_noz",
        mlp_ratio=4.0,
    ),
    "vmamba_base": dict(
        depths=[2, 2, 15, 2],
        dims=128,
        drop_path_rate=0.6,
        ssm_d_state=1,
        ssm_ratio=2.0,
        forward_type="v05_noz",
        mlp_ratio=4.0,
    ),
}


@MODELS.register_module()
class VMambaBackbone(BaseModule):
    arch_settings = VMAMBA_CONFIGS

    def __init__(
        self,
        variant="vmamba_tiny",
        pretrained=None,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(init_cfg=init_cfg)
        if variant not in self.arch_settings:
            raise ValueError(f"Unknown VMamba variant: {variant}. "
                             f"Available: {list(self.arch_settings.keys())}")

        cfg = self.arch_settings[variant]
        self.model = Backbone_VSSM(
            pretrained=pretrained,
            patch_size=4,
            in_chans=3,
            num_classes=1000,
            ssm_dt_rank="auto",
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=False,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            gmlp=False,
            patch_norm=True,
            norm_layer="ln2d",
            downsample_version="v3",
            patchembed_version="v2",
            use_checkpoint=False,
            posembed=False,
            imgsize=256,
            **cfg,
        )
        self.out_indices = (0, 1, 2, 3)
        self._out_channels = self.model.dims

    def forward(self, x):
        features_dict = self.model(x)
        return tuple(features_dict[f"stage{i+1}"] for i in range(4))

    @property
    def out_channels(self):
        return self._out_channels
