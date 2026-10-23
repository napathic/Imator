import torch.nn.functional as F
from torch import nn
import torch

from model.common import Transformer, Patchifier

class Predictor(nn.Module):
  def __init__(self,
                n_layers: int,
                n_heads: int,
                head_dim: int,
                hidden_dim: int,
                proj_dim: int,
                image_height: int,
                image_width: int,
                bias: bool = True
              ):
    super().__init__()
    self.model = Transformer(n_layers, n_heads, head_dim, hidden_dim, proj_dim, bias)
    self.patcher = Patchifier(2, image_height, image_width)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    pass