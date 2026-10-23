import torch.nn.functional as F
from torch import nn
import torch
import math


class Attention(nn.Module):
  def __init__(self,
                n_heads: int,
                head_dim: int,
                hidden_dim: int,
                bias: bool = True
              ):
    super().__init__()
    assert n_heads * head_dim == hidden_dim, "n_heads * head_dim shoud be hidden_dim."
    self.gqkv = nn.Linear(hidden_dim, hidden_dim * 4, bias=bias)
    self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
    self.head_dim = head_dim
    self.n_heads = n_heads
    
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    gate, q, k, v = [
      t.unflatten(-1, (self.n_heads, self.head_dim)).transpose(-3, -2)
        for t in self.gqkv(x).chunk(4, dim=-1)
    ]

    attn = F.scaled_dot_product_attention(q, k, v) * F.silu(gate)
    attn = attn.transpose(-3, -2).flatten(-2)
    return self.out_proj(attn)

class Mlp(nn.Module):
  def __init__(self,
                hidden_dim: int,
                proj_dim: int,
                bias: bool = True
              ):
    super().__init__()
    self.gup = nn.Linear(hidden_dim, proj_dim * 2, bias=bias)
    self.down = nn.Linear(proj_dim, hidden_dim, bias=bias)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    gate, up = self.gup(x).chunk(2, dim=-1)
    act = up * F.silu(gate)
    return self.down(act)

class TransformerLayer(nn.Module):
  def __init__(self,
                n_heads: int,
                head_dim: int,
                hidden_dim: int,
                proj_dim: int,
                bias: bool = True
              ):
    super().__init__()
    self.attn = Attention(n_heads, head_dim, hidden_dim, bias)
    self.attn_norm = nn.RMSNorm(hidden_dim)

    self.mlp = Mlp(hidden_dim, proj_dim, bias)
    self.mlp_norm = nn.RMSNorm(hidden_dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x + self.attn(self.attn_norm(x))
    return x + self.mlp(self.mlp_norm(x))

class Transformer(nn.Module):
  def __init__(self,
                n_layers: int,
                n_heads: int,
                head_dim: int,
                hidden_dim: int,
                proj_dim: int,
                bias: bool = True
              ):
    super().__init__()
    self.layers = nn.ModuleList([
      TransformerLayer(n_heads, head_dim, hidden_dim, proj_dim, bias)
        for _layer_idx in range(n_layers)
    ])

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    for layer in self.layers:
      x = layer(x)
    return x

class Patchifier(nn.Module):
  def __init__(self, patch_size: int, height: int | None = None, width: int | None = None):
    super().__init__()
    self.p_s = patch_size
    self.height = height
    self.width = width

  def patchify(self, x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    return (x
            .reshape(b, c, h // self.p_s, self.p_s, w // self.p_s, self.p_s)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(b, h * w // self.p_s ** 2, c * self.p_s ** 2)
          )

  def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
    b, t, d = x.shape
    p = self.p_s
    c = d // p ** 2
    gh, gw = (
      (self.height // p, self.width // p)
      if self.height and self.width
      else (math.isqrt(t), math.isqrt(t))
    )
    assert gh * gw == t, f"expected ({gh} * {gw} == tokens), got {t} tokens"
    return (x
            .reshape(b, gh, gw, p, p, c)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(b, c, gh * p, gw * p)
          )