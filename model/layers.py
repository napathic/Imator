import torch.nn.functional as F
from torch import nn
import torch

import math


class DownResidualBlock(nn.Module):
  def __init__(self,
                in_channels: int,
                scaler: int,
                out_channels: int,
                kernel_size: int,
                num_groups: int
              ):
    super().__init__()

    self.main = nn.Sequential(
      nn.Conv2d(
        in_channels,
        scaler,
        kernel_size,
        padding=1
      ),
      nn.GroupNorm(num_groups, scaler),
      nn.SiLU(),

      nn.Conv2d(
        scaler,
        out_channels,
        kernel_size,
        stride=2,
        padding=1
      ),
      nn.GroupNorm(num_groups, out_channels),
    )

    self.skip = nn.Conv2d(
      in_channels,
      out_channels,
      kernel_size=1,
      stride=2,
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    y = self.main(x)
    residual = self.skip(x)
    return F.silu(y + residual)


class UpResidualBlock(nn.Module):
  def __init__(self,
                in_channels: int,
                scaler: int,
                out_channels: int,
                kernel_size: int,
                num_groups: int
              ):
    super().__init__()

    self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
    self.main = nn.Sequential(
      nn.Conv2d(
        in_channels,
        scaler,
        kernel_size,
        padding=kernel_size // 2,
      ),
      nn.GroupNorm(num_groups, scaler),
      nn.SiLU(),

      nn.Conv2d(
        scaler,
        out_channels,
        kernel_size,
        padding=kernel_size // 2,
      ),
      nn.GroupNorm(num_groups, out_channels),
    )

    self.skip = nn.Conv2d(
      in_channels,
      out_channels,
      kernel_size=1,
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.upsample(x)
    y = self.main(x)

    return F.silu(y + self.skip(x))


def get_layers(
    in_channels: int,
    kernel_size: int,
    start_channels: int = 16,
    layers: int = 3,
    decoder: bool = False,
) -> list[nn.Module]:
  output = []

  block_type = UpResidualBlock if decoder else DownResidualBlock
  layer_indices = reversed(range(layers)) if decoder else range(layers)

  for layer_idx in layer_indices:
    i = layer_idx * 2

    encoder_input = (in_channels if layer_idx == 0 else i * start_channels)
    scaler = (i + 1) * start_channels
    encoder_output = (i + 2) * start_channels

    block_output = encoder_input if decoder else encoder_output
    output.append(
      block_type(
        in_channels=encoder_output if decoder else encoder_input,
        scaler=scaler,
        out_channels=block_output,
        kernel_size=kernel_size,
        num_groups=math.gcd(scaler, block_output),
      )
    )

  return output
