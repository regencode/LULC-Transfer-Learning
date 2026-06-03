"""MambaVision backbone wrapper for MMSegmentation.

Wraps the existing MambaVision model to conform to mmseg's backbone interface:
forward() returns a tuple of multi-scale feature tensors.
"""

from mmengine.model import BaseModule
from mmseg.registry import MODELS

try:
    from .MambaVisionModels.mamba_vision import (
        mamba_vision_T,
        mamba_vision_T2,
        mamba_vision_S,
        mamba_vision_B,
        mamba_vision_L,
    )
    MAMBAVISION_AVAILABLE = True
except ImportError:
    MAMBAVISION_AVAILABLE = False

MAMBAVISION_FACTORIES = {
    "mambavision_t": mamba_vision_T,
    "mambavision_t2": mamba_vision_T2,
    "mambavision_s": mamba_vision_S,
    "mambavision_b": mamba_vision_B,
    "mambavision_l": mamba_vision_L,
}


@MODELS.register_module()
class MambaVisionBackbone(BaseModule):
    arch_factories = MAMBAVISION_FACTORIES

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
        if variant not in self.arch_factories:
            raise ValueError(f"Unknown MambaVision variant: {variant}. "
                             f"Available: {list(self.arch_factories.keys())}")

        factory = self.arch_factories[variant]
        if pretrained:
            self.model = factory(pretrained=True, model_path=pretrained)
        else:
            self.model = factory(pretrained=False)

        self.out_indices = (0, 1, 2, 3)
        self._out_channels = self.model.get_stage_channels()

    def forward(self, x):
        features_dict = self.model(x)
        return tuple(features_dict[f"stage{i+1}"] for i in range(4))

    @property
    def out_channels(self):
        return self._out_channels
