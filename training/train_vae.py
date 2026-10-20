from dataclasses import asdict
from model.autoencoder import Vae
from tqdm import tqdm

from data.ImageClass import OpenImagesDataset
from torch.utils.data import DataLoader
from pathlib import Path

from config import cfg
import torch

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")


def main() -> None:
  train_set = OpenImagesDataset(
    image_dir=Path("data") / "open-images" / f"train_images_{cfg.image.image_size}",
    image_size=cfg.image.image_size
  )

  train_loader = DataLoader(
    dataset=train_set,
    batch_size=cfg.training.batch_size,
    shuffle=True,
    num_workers=4,
    persistent_workers=True,
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
    eta_min=cfg.training.vae_lr * 0.7
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
    for step, x in enumerate(loop):
      x: torch.Tensor = x.to(cfg.device)
      pred, mu, logvar = vae(x)
      loss = vae.reconstruction_loss(reconstruction=pred, target=x, logvar=logvar, mu=mu)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      scheduler.step()

      if step % 20 == 0:
        loss_value = loss.item()
        ema_loss = loss_value if ema_loss is None else loss_value * ema_alpha + (1 - ema_alpha) * ema_loss
        loop.set_postfix(lr=f"{scheduler.get_last_lr()[0]:.2e}", loss=ema_loss)
    ckpt_file = folder / f"vae{epoch + 1}.pt"
    ckpt_file.parent.mkdir(parents=True, exist_ok=True)

    temp_file = ckpt_file.with_suffix(".pt.tmp")
    torch.save({
      "epoch": epoch + 1,
      "model_state_dict": vae.state_dict(),
      "optimizer_state_dict": optimizer.state_dict(),
      "scheduler_state_dict": scheduler.state_dict(),
      "ema_loss": ema_loss,
    }, temp_file)
    temp_file.replace(ckpt_file)


if __name__ == "__main__":
  main()
