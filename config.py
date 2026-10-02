from omegaconf import OmegaConf
from dataclasses import dataclass, field
import torch

from pathlib import Path

@dataclass
class Paths:
  base_dir: Path = field(default=Path(__file__).resolve().parent)
  run_dir: Path = field(default_factory=Path.cwd)

  checkpoint_folder: str = "ckpt"
  training_folder: str = "training"
  test_folder: str = "tests"

  @property
  def ckpt_dir(self) -> Path:
    return self.base_dir / self.checkpoint_folder

  @property
  def training_dir(self) -> Path:
    return self.base_dir / self.training_folder

  @property
  def test_dir(self) -> Path:
    return self.base_dir / self.test_folder

@dataclass
class Image:
  channels: int = 3
  image_size: int = 32

@dataclass
class Vae:
  out_channels: int = 8
  in_channels: int = Image.channels
  kernel_size: int = 3
  start_channels: int = 32
  layers: int = 4
  downsample_steps: int = 3

@dataclass
class Training:
  vae_lr: float = 3e-4
  batch_size: int = 64
  epochs: int = 10

@dataclass
class Config():
  paths: Paths = field(default_factory=Paths)
  vae: Vae = field(default_factory=Vae)
  training: Training = field(default_factory=Training)
  image: Image = field(default_factory=Image)

  device: str = field(
    default_factory=lambda: (
      "cuda" if torch.cuda.is_available() else
      "mps" if torch.backends.mps.is_available()
      else "cpu"
    ))


structured_cfg = OmegaConf.structured(Config)
cfg: Config = OmegaConf.to_object(structured_cfg)
