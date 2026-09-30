import torch.nn.functional as F
from torch import nn
import torch


def Conv2d(
        in_channels: int,
        out_channels: int,
        kernal_size: int = 3,
        stride: int = 2,
        padding: int = 1
      ):
  return nn.Conv2d(in_channels, out_channels, kernal_size, stride, padding)

def get_args(
          in_channels: int,
          start_channels: int = 16,
          layers: int = 3
        ) -> list:
  out = []
  for _layer_idx in range(0, layers):
    i = _layer_idx * 2
    scaler = (i + 1) * start_channels
    out_layer = (i + 2) * start_channels

    out.extend([
      Conv2d(
          in_channels if i == 0 else i * start_channels,
          scaler, stride=1
        ),
      nn.GroupNorm(8, scaler),
      nn.SiLU(),
      Conv2d(
        scaler,
        out_layer
      ),
      nn.GroupNorm(8, out_layer),
      nn.SiLU(),
    ])
  return out

class Encoder(nn.Module): # [B, C, H, W]
  def __init__(self,
              out_channels: int,
              in_channels: int = 3,
              kernel_size: int = 3,
              start_channels: int = 16,
              layers: int = 4
            ):
    super().__init__()

    self.backbone = nn.Sequential(
      *get_args(
        in_channels=in_channels,
        start_channels=start_channels,
        layers=layers
      )
    )
    _out_backbone_dim = start_channels * layers * 2

    self.to_logvar = nn.Conv2d(_out_backbone_dim, out_channels, kernel_size)
    self.to_mu = nn.Conv2d(_out_backbone_dim, out_channels, kernel_size)

  def forward(self, x: torch.Tensor) -> list[torch.Tensor, torch.Tensor]:
    features = self.backbone(x)
    return [
      self.to_mu(features),
      self.to_logvar(features)
    ]