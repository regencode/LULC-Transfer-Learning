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
    from .VMambaModels.vmamba import vmamba_small_s2l15, vmamba_base_s2l15, vmamba_tiny_s1l8
    VMAMBA_AVAILABLE = True
except ImportError:
    VMAMBA_AVAILABLE = False
    vmamba_base_s2l15 = None
    vmamba_small_s2l15 = None
    vmamba_tiny_s1l8 = None


@register_backbone("vmamba_tiny")
def vmamba_tiny_backbone(pretrained: bool = True):
    if not VMAMBA_AVAILABLE: 
        print("VMamba is not available")
        return None
    assert vmamba_tiny_s1l8 is not None
    if pretrained: return vmamba_tiny_s1l8(pretrained="/tmp/vssm1_tiny_0230s_ckpt_epoch_264.pth")
    return vmamba_tiny_s1l8()

@register_backbone("vmamba_small")
def vmamba_small_backbone(pretrained: bool = True):
    if not VMAMBA_AVAILABLE: 
        print("VMamba is not available")
        return None
    assert vmamba_small_s2l15 is not None
    if pretrained: return vmamba_small_s2l15(pretrained="/tmp/vssm_small_0229_ckpt_epoch_222.pth")
    return vmamba_small_s2l15()

@register_backbone("vmamba_base")
def vmamba_base_backbone(pretrained: bool = True):
    if not VMAMBA_AVAILABLE: 
        print("VMamba is not available")
        return None
    assert vmamba_base_s2l15 is not None
    if pretrained: return vmamba_base_s2l15(pretrained="/tmp/vssm_base_0229_ckpt_epoch_237.pth")
    return vmamba_base_s2l15()
