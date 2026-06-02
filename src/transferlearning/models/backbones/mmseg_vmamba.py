"""VMamba backbone wrapper for MMSegmentation.

Wraps the existing Backbone_VSSM from VMambaModels to conform to mmseg's
backbone interface: forward() returns a tuple of multi-scale feature tensors.
"""

import torch
from mmengine.model import BaseModule
from mmseg.registry import MODELS
from pathlib import Path

from .VMambaModels.vmamba import vmamba_small_s2l15, vmamba_base_s2l15

VMAMBA_FACTORIES = {
    "vmamba_small": vmamba_small_s2l15,
    "vmamba_base": vmamba_base_s2l15,
}

VMAMBA_URLS = {
    "vmamba_small": "https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm_small_0229_ckpt_epoch_222.pth",
    "vmamba_base": "https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm_base_0229_ckpt_epoch_237.pth",
}


@MODELS.register_module()
class VMambaBackbone(BaseModule):
    arch_factories = VMAMBA_FACTORIES

    def __init__(
        self,
        variant="vmamba_small",
        pretrained=None,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(init_cfg=init_cfg)
        if variant not in self.arch_factories:
            raise ValueError(f"Unknown VMamba variant: {variant}. "
                             f"Available: {list(self.arch_factories.keys())}")

        self.model = self.arch_factories[variant](pretrained="")
        if pretrained:
            self._load_pretrained(pretrained, VMAMBA_URLS.get(variant))
        self.out_indices = (0, 1, 2, 3)
        self._out_channels = self.model.dims

    def _load_pretrained(self, path, url=None):
        if not Path(path).is_file() and url:
            torch.hub.download_url_to_file(url=url, dst=path)
        self.model.load_pretrained(path)

    def forward(self, x):
        features_dict = self.model(x)
        return tuple(features_dict[f"stage{i+1}"] for i in range(4))

    @property
    def out_channels(self):
        return self._out_channels
