from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("UNet depth must be >= 2.")

        encoder_channels = [base_channels * (2**level) for level in range(depth)]
        self.down_blocks = nn.ModuleList()
        previous = in_channels
        for channels in encoder_channels:
            self.down_blocks.append(ConvBlock(previous, channels, dropout=dropout))
            previous = channels

        self.up_transpose = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for level in range(depth - 2, -1, -1):
            in_ch = encoder_channels[level + 1]
            out_ch = encoder_channels[level]
            self.up_transpose.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2))
            self.up_blocks.append(ConvBlock(out_ch * 2, out_ch, dropout=dropout))

        self.output = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        for block in self.down_blocks[:-1]:
            x = block(x)
            skips.append(x)
            x = F.max_pool2d(x, kernel_size=2)

        x = self.down_blocks[-1](x)

        for up, block, skip in zip(self.up_transpose, self.up_blocks, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = block(x)
        return self.output(x)


def build_model(config: dict) -> UNet:
    return UNet(
        in_channels=int(config.get("in_channels", 1)),
        out_channels=int(config.get("out_channels", 1)),
        base_channels=int(config.get("base_channels", 32)),
        depth=int(config.get("depth", 4)),
        dropout=float(config.get("dropout", 0.0)),
    )

