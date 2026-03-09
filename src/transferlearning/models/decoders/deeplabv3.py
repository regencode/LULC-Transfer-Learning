from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_decoder


class ASPPConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ASPPPooling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]
        x = self.pool(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: List[int], out_channels: int = 256):
        super().__init__()
        modules = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        ]
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = [conv(x) for conv in self.convs]
        return self.project(torch.cat(res, dim=1))


@register_decoder("deeplabv3plus")
class DeepLabV3PlusDecoder(nn.Module):
    """DeepLabV3+ decoder with ASPP and low-level feature fusion.

    Uses stage4 features for ASPP and stage1 features as low-level skip.
    Expects backbone features as a dict with keys stage1..stage4.
    """

    def __init__(
        self,
        num_classes: int,
        stage_channels: List[int],
        aspp_out_channels: int = 256,
        low_level_channels: int = 48,
        atrous_rates: List[int] = None,
    ):
        super().__init__()
        if atrous_rates is None:
            atrous_rates = [6, 12, 18]

        c1 = stage_channels[0]
        c4 = stage_channels[-1]

        self.aspp = ASPP(c4, atrous_rates, aspp_out_channels)

        self.low_level_conv = nn.Sequential(
            nn.Conv2d(c1, low_level_channels, 1, bias=False),
            nn.BatchNorm2d(low_level_channels),
            nn.ReLU(inplace=True),
        )

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(aspp_out_channels + low_level_channels, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, features: dict) -> torch.Tensor:
        low_level = self.low_level_conv(features["stage1"])
        high_level = self.aspp(features["stage4"])
        high_level = F.interpolate(high_level, size=low_level.shape[2:], mode="bilinear", align_corners=False)
        fused = torch.cat([high_level, low_level], dim=1)
        fused = self.fuse_conv(fused)
        return self.head(fused)
