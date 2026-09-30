from omegaconf import OmegaConf
from dataclasses import dataclass, field

from pathlib import Path

@dataclass
class Paths:
  base_dir: Path = field(default=Path(__file__).resolve().parent)
  run_dir = field(default_factory=Path.cwd())

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

