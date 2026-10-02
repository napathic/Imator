import torch.nn.functional as F
from torch import nn
import torch

from dataclasses import asdict
from model.autoencoder import Vae
from tqdm import tqdm

from data.ImageClass import OpenImagesDataset
from torch.utils.data import DataLoader
from pathlib import Path

from config import cfg

train_set = OpenImagesDataset(
  image_dir=Path("data") / "open-images" / "images",
  image_size=32
)

train_loader = DataLoader(
  dataset=train_set,
  batch_size=cfg.training.batch_size,
  shuffle=True
)

vae = Vae(**asdict(cfg.vae))
vae.to(cfg.device)


