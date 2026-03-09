from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_decoder


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


@register_decoder("unet")
class UNetDecoder(nn.Module):
    """U-Net decoder that consumes multi-scale backbone features via skip connections.

    Expects backbone features as a dict with keys stage1..stage4,
    each a (B, C, H, W) tensor at progressively lower resolutions.
    stage_channels must match the backbone's get_stage_channels() output.
    """

    def __init__(self, num_classes: int, stage_channels: List[int]):
        super().__init__()
        c1, c2, c3, c4 = stage_channels

        self.up4 = UpBlock(c4, c3, c3)
        self.up3 = UpBlock(c3, c2, c2)
        self.up2 = UpBlock(c2, c1, c1)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(c1, c1, kernel_size=2, stride=2),
            ConvBlock(c1, c1),
        )
        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, features: dict) -> torch.Tensor:
        x = features["stage4"]
        x = self.up4(x, features["stage3"])
        x = self.up3(x, features["stage2"])
        x = self.up2(x, features["stage1"])
        x = self.final_up(x)
        return self.head(x)
