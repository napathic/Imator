from omegaconf import OmegaConf
from dataclasses import dataclass, field

from pathlib import Path

@dataclass
class Paths:
  base_dir: Path = field(default=Path(__file__).resolve().parent)
  run_dir: Path = field(default_factory=Path.cwd)

  checkpoint_folder: str = "ckpt"
  training_folder: str = "training"
  test_folder: str = "tests"

  @property
  def checkpoint_dir(self) -> Path:
    return self.base_dir / self.checkpoint_folder

  @property
  def training_dir(self) -> Path:
    return self.base_dir / self.training_folder

  @property
  def test_dir(self) -> Path:
    return self.base_dir / self.test_folder

@dataclass
class Vae:
  out_channels: int = 8
  in_channels: int = 3
  kernel_size: int = 3
  start_channels: int = 20
  layers: int = 4

@dataclass
class Config():
  paths: Paths = field(default_factory=Paths)
  vae: Vae = field(default_factory=Vae)

cfg: Config = OmegaConf.structured(Config)