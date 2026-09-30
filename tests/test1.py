from model.autoencoder import Vae
import torch

from config import cfg

vae: Vae = Vae(**cfg.vae)

input = (2, 3, 64, 64)
r = torch.randn(*input)

out: torch.Tensor = vae(r)
print(out.shape)

assert r.shape == out.shape, f"Wrong output shape, got {out.shape} expected {r.shape}"