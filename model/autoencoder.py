import torch.nn.functional as F
from torch import nn
import torch

import lpips
from .layers import get_layers


class Encoder(nn.Module):
  def __init__(self,
                out_channels: int,
                in_channels: int = 3,
                kernel_size: int = 3,
                start_channels: int = 16,
                layers: int = 4,
                downsample_steps: int = 3,
              ):
    super().__init__()

    self.backbone = nn.Sequential(
      *get_layers(
        in_channels=in_channels,
        kernel_size=kernel_size,
        start_channels=start_channels,
        layers=layers,
        decoder=False,
        downsample_steps=downsample_steps,
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
                layers: int = 4,
                downsample_steps: int = 3,
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
        decoder=True,
        downsample_steps=downsample_steps,
        feature_output=True,
      )
    )
    self.to_image = nn.Conv2d(start_channels, in_channels, kernel_size, padding=kernel_size // 2)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    features = self.from_latent(x)
    return self.to_image(self.backbone(features))

class Vae(nn.Module):
  _MS_SSIM_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)

  def __init__(self,
                out_channels: int,
                in_channels: int = 3,
                kernel_size: int = 3,
                start_channels: int = 16,
                layers: int = 4,
                downsample_steps: int = 3,
              ):
    super().__init__()
    params = (
      out_channels,
      in_channels,
      kernel_size,
      start_channels,
      layers,
      downsample_steps
    )

    self.encoder: Encoder = Encoder(*params)
    self.decoder: Decoder = Decoder(*params)
    self._lpips = {}

  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu, logvar = self.encoder(x)
    latent = self.reparameterize(mu, logvar) if self.training else mu
    return self.decoder(latent), mu, logvar

  # helpers
  def _get_lpips(self, device: torch.device) -> lpips.LPIPS:
    net = 'vgg' if self.training else 'alex'
    if net not in self._lpips:
      fn = lpips.LPIPS(net=net, verbose=False).eval()
      for p in fn.parameters():
        p.requires_grad_(False)
      self._lpips[net] = fn
    return self._lpips[net].to(device)

  @staticmethod
  def _gaussian_window(
      size: int,
      sigma: float,
      channels: int,
      like: torch.Tensor
    ) -> torch.Tensor:
    coords = torch.arange(size, dtype=like.dtype, device=like.device) - size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    w = g[:, None] * g[None, :]
    return w.expand(channels, 1, size, size).contiguous()
  

  @staticmethod
  def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

  # loss's
  def reconstruction_loss(self,
                          reconstruction: torch.Tensor,
                          target: torch.Tensor,
                          logvar: torch.Tensor,
                          mu: torch.Tensor
                        ) -> torch.Tensor:
    l1_loss = F.l1_loss(reconstruction, target)
    p_loss = self._get_lpips(target.device)(reconstruction, target, normalize=True).mean()
    kl_loss = self._KL_LOSS(logvar, mu)
    s_loss = self.ms_ssim_loss(reconstruction, target)
    return 0.7 * s_loss + 0.4 * l1_loss + 0.6 * p_loss + 1e-6 * kl_loss

  @staticmethod
  @staticmethod
  def _ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    window: torch.Tensor,
    C1: float, C2: float
  ) -> tuple[torch.Tensor, torch.Tensor]:
    C = x.shape[1]
    blur = lambda t: F.conv2d(t, window, groups=C)
    mu_x, mu_y = blur(x), blur(y)
    var_x = blur(x * x) - mu_x ** 2
    var_y = blur(y * y) - mu_y ** 2
    cov = blur(x * y) - mu_x * mu_y
    l = (2 * mu_x * mu_y + C1) / (mu_x ** 2 + mu_y ** 2 + C1)
    cs = (2 * cov + C2) / (var_x + var_y + C2)
    return (l * cs).flatten(1).mean(1), cs.flatten(1).mean(1)

  def ms_ssim_loss(self,
                    x: torch.Tensor,
                    y: torch.Tensor, 
                    data_range: float = 1.0,
                    win: int = 11,
                    sigma: float = 1.5
                  ) -> torch.Tensor:
    # use only the scales where the window still fits, small images can't do all 5
    min_side = min(x.shape[-2:])
    scales = 1
    while scales < len(self._MS_SSIM_WEIGHTS) and min_side // 2 ** scales >= win:
      scales += 1
    weights = torch.tensor(self._MS_SSIM_WEIGHTS[:scales], device=x.device, dtype=x.dtype)
    weights = weights / weights.sum()
    window = self._gaussian_window(win, sigma, x.shape[1], x)
    C1, C2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2

    vals = []
    for i in range(len(weights)):
      ssim, cs = self._ssim(x, y, window, C1, C2)
      if i < len(weights) - 1:
        vals.append(F.relu(cs))
        x, y = F.avg_pool2d(x, 2), F.avg_pool2d(y, 2)
    vals.append(F.relu(ssim))

    vals = torch.stack(vals)
    ms = torch.prod(vals ** weights[:, None], dim=0)
    return 1 - ms.mean()

  @staticmethod
  def _KL_LOSS(logvar: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return kl.sum(dim=(1, 2, 3)).mean()
