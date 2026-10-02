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

folder = cfg.paths.ckpt_dir / "vae"
n_epoch = lambda path: int(path.stem.removeprefix("vae"))

latest = max(
  (path for path in folder.glob("vae*.pt") if path.stem.removeprefix("vae").isdigit()),
  key=n_epoch,
  default=None,
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
start_epoch = 0
if latest is not None:
  checkpoint = torch.load(latest, map_location=cfg.device, weights_only=True)
  vae.load_state_dict(checkpoint["model_state_dict"])
  optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
  scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
  start_epoch = checkpoint["epoch"]
  ema_loss = checkpoint.get("ema_loss")


for epoch in range(start_epoch, cfg.training.epochs):
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
  ckpt_file = folder / f"vae{epoch + 1}.pt"
  ckpt_file.parent.mkdir(parents=True, exist_ok=True)
  # Publish only complete epoch checkpoints, so interrupted saves aren't loaded.
  temp_file = ckpt_file.with_suffix(".pt.tmp")
  torch.save({
    "epoch": epoch + 1,
    "model_state_dict": vae.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "ema_loss": ema_loss,
  }, temp_file)
  temp_file.replace(ckpt_file)
