import torch.nn.functional as F
from torch import nn
import torch

from .layers import get_layers


class Encoder(nn.Module):
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
        layers=layers,
        decoder=False
      )
    )
    _out_backbone_dim = start_channels * layers * 2

    self.to_logvar = nn.Conv2d(_out_backbone_dim, out_channels, kernel_size, padding=kernel_size // 2)
    self.to_mu = nn.Conv2d(_out_backbone_dim, out_channels, kernel_size, padding=kernel_size // 2)

  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    features = self.backbone(x)
    return (
      self.to_mu(features),
      self.to_logvar(features)
    )

class Decoder(nn.Module):
  def __init__(self,
                out_channels: int,
                in_channels: int = 3,
                kernel_size: int = 3,
                start_channels: int = 16,
                layers: int = 4
              ):
    super().__init__()


    _out_backbone_dim = start_channels * layers * 2
    self.from_latent = nn.Conv2d(
                      out_channels,
                      _out_backbone_dim,
                      kernel_size=3,
                      padding=1,
                    )

    self.backbone = nn.Sequential(
      *get_layers(
        in_channels=in_channels,
        kernel_size=kernel_size,
        start_channels=start_channels,
        layers=layers,
        decoder=True
      )
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    features = self.from_latent(x)
    return self.backbone(features)

class Vae(nn.Module):
  def __init__(self,
                out_channels: int,
                in_channels: int = 3,
                kernel_size: int = 3,
                start_channels: int = 16,
                layers: int = 4
              ):
    super().__init__()
    params = (out_channels, in_channels, kernel_size, start_channels, layers)

    self.encoder: Encoder = Encoder(*params)
    self.decoder: Decoder = Decoder(*params)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    mu, logvar = self.encoder(x)
    latent = self.reparameterize(mu, logvar)
    return self.decoder(latent)

  @staticmethod
  def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

  @staticmethod
  def reconstruction_loss(reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(reconstruction, target)# + ... # NOTE: start with simple loss, make sure its working, then add MS-SSIN + l1 loss

  @staticmethod
  def _SSIM_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    ...