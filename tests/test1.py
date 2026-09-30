from model.autoencoder import Encoder
import torch

encoder: Encoder = Encoder(out_channels=8)
r = torch.randn(2, 3, 64, 64)

mu, logvar = encoder(r)

print(logvar.shape)
print(mu.shape)