from model.autoencoder import Encoder
import torch

from config import cfg

encoder: Encoder = Encoder(**cfg.vae)
input = (2, 3, 64, 64)
r = torch.randn(*input)

mu, logvar = encoder(r)

print(logvar.shape)
print(mu.shape)

for x in [mu, logvar]:
  assert x.shape[-3] == cfg.vae.out_channels, f"Mismatch in shapes, got shape {x.shape[-3]} Expected {cfg.vae.out_channels}"
  assert x.shape[-1] == (input[-1] / ((cfg.vae.layers ** 2) * 2)), f"Mismatch in shapes, got shape {x.shape[-1]}, Expected {input[-1] / ((cfg.vae.layers ** 2) * 2)}"
  assert x.shape[-1] == x.shape[-2]
  assert x.shape[0] == input[0], "Wrong batch dim"