import torch.nn.functional as F
from torch import nn
import torch


class DownResidualBlock(nn.Module):
    def __init__(self, in_channels: int, scaler: int, out_channels: int, kernel_size: int, group_size: int):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(
                in_channels,
                scaler,
                kernel_size,
                padding=1
              ),
            nn.GroupNorm(group_size, scaler),
            nn.SiLU(),

            nn.Conv2d(
                scaler,
                out_channels,
                kernel_size,
                stride=2,
                padding=1
              ),
            nn.GroupNorm(group_size, out_channels),
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

def get_layers(
          in_channels: int,
          kernel_size: int,
          start_channels: int = 16,
          layers: int = 3
        ) -> list:
  out = []
  for _layer_idx in range(0, layers):
    i = _layer_idx * 2
    scaler = (i + 1) * start_channels
    out_layer = (i + 2) * start_channels

    out.append(
       DownResidualBlock(
          in_channels if i == 0 else i * start_channels,
          scaler,
          out_channels=out_layer,
          kernel_size=kernel_size,
          group_size=start_channels
        )
      )
  return out