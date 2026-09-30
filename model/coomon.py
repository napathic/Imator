import torch.nn.functional as F
from torch import nn
import torch

class Attention(nn.Module):
  def __init__(self,
              n_heads: int,
              head_dim: int,
              hidden_dim: int,
              bias: bool = True,
            ):
    super().__init__()
    assert n_heads * head_dim == hidden_dim, "n_heads * head_dim must be hidden_dim!"
    self.gqkv = nn.Linear(hidden_dim, hidden_dim * 4, bias=bias)
    self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
    self.n_heads = n_heads
    self.head_dim = head_dim

  def forward(self, x: torch.Tensor) -> torch.Tensor: # [b, seq, n_h, h_d]
    gate, q, k, v = [
      f.unflatten(-1, (self.n_heads, self.head_dim)).transpose(-3, -2)
        for f in self.gqkv(x).chunk(4, dim=-1)
    ]

    attn: torch.Tensor = F.scaled_dot_product_attention(q, k, v) * F.silu(gate)
    attn = attn.transpose(-3, -2).flatten(-2)
    return self.out_proj(attn)