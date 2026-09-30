import torch.nn.functional as F
from torch import nn
import torch

from .layers import get_layers


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
      *get_layers(
        in_channels=in_channels,
        kernel_size=kernel_size,
        start_channels=start_channels,
        layers=layers
      )
    )
    _out_backbone_dim = start_channels * layers * 2

    self.to_logvar = nn.Conv2d(_out_backbone_dim, out_channels, kernel_size)
    self.to_mu = nn.Conv2d(_out_backbone_dim, out_channels, kernel_size)

  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    features = self.backbone(x)
    return (
      self.to_mu(features),
      self.to_logvar(features)
    )