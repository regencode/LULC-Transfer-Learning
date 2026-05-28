"""UNet decode head for MMSegmentation.

Adapted from the existing UNetDecoder to conform to mmseg's BaseDecodeHead
interface. Uses skip connections from all 4 backbone stages.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


@MODELS.register_module()
class UNetHead(BaseDecodeHead):
    def __init__(self, **kwargs):
        super().__init__(input_transform="multiple_select", **kwargs)
        c1, c2, c3, c4 = self.in_channels

        self.up4 = UpBlock(c4, c3, c3)
        self.up3 = UpBlock(c3, c2, c2)
        self.up2 = UpBlock(c2, c1, c1)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(c1, c1, kernel_size=2, stride=2),
            ConvBlock(c1, c1),
        )

    def forward(self, inputs):
        s1, s2, s3, s4 = inputs

        x = self.up4(s4, s3)
        x = self.up3(x, s2)
        x = self.up2(x, s1)
        x = self.final_up(x)
        output = self.cls_seg(x)
        return output
