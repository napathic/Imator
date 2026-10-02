from dataclasses import asdict
from model.autoencoder import Vae

from data.ImageClass import OpenImagesDataset
from pathlib import Path
from PIL import Image
import torch

from config import cfg

train_set = OpenImagesDataset(
  image_dir=Path("data") / "open-images" / "test_images",
  image_size=cfg.image.image_size
)

vae = Vae(**asdict(cfg.vae))
vae.to(cfg.device)


for x in train_set:
  x: torch.Tensor = x.unsqueeze(0).to(cfg.device)
  pred: torch.Tensor = vae(x).squeeze(0)

  torch_pixels = torch.cat((x[0], pred), dim=2)

  pixels = (
    torch_pixels
    .detach()
    .cpu()
    .clamp(0, 1)
    .permute(1, 2, 0)
    .mul(255)
    .round()
    .to(torch.uint8)
    .numpy()
  )

  Image.fromarray(pixels).show()

  z = input("\"q\" to exit.")
  if z.lower() == "q":
    break
