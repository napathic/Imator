from dataclasses import asdict
from model.autoencoder import Vae
from tqdm import tqdm

from data.ImageClass import OpenImagesDataset
from torch.utils.data import DataLoader
from pathlib import Path

from config import cfg
import torch


train_set = OpenImagesDataset(
  image_dir=Path("data") / "open-images" / "train_images",
  image_size=cfg.image.image_size
)

train_loader = DataLoader(
  dataset=train_set,
  batch_size=cfg.training.batch_size,
  shuffle=True
)

vae = Vae(**asdict(cfg.vae))
vae.to(cfg.device)

optimizer = torch.optim.AdamW(params=vae.parameters(), lr=cfg.training.vae_lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
  optimizer=optimizer,
  T_max=cfg.training.epochs * len(train_loader),
  eta_min=cfg.training.vae_lr * 0.4
)

ema_loss = None
ema_alpha = 0.05
for epoch in range(cfg.training.epochs):
  loop = tqdm(iterable=train_loader, desc=f"Epoch {epoch + 1}/{cfg.training.epochs}")
  for x in loop:
    x: torch.Tensor = x.to(cfg.device)
    pred = vae(x)
    loss = vae.reconstruction_loss(reconstruction=pred, target=x)

    optimizer.zero_grad()
    loss.backward()
    ema_loss = loss.item() if ema_loss is None else loss.item() * ema_alpha + (1 - ema_alpha) * ema_loss

    optimizer.step()
    scheduler.step()

    loop.set_postfix(lr=f"{scheduler.get_last_lr()[0]:.2e}", loss=ema_loss)
  ckpt_file = cfg.paths.ckpt_dir / "vae" / f"vae{epoch + 1}.pt"
  ckpt_file.parent.mkdir(parents=True, exist_ok=True)
  torch.save(vae.state_dict(), ckpt_file)
