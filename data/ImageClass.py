from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class OpenImagesDataset(Dataset):
  def __init__(self,
              image_dir: str,
              image_size: int = 64,
            ):
    self.image_paths = sorted(
      path for path in Path(image_dir).iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    self.transform = transforms.Compose([
      transforms.Resize(
        image_size,
        interpolation=transforms.InterpolationMode.BILINEAR,
        antialias=True,
      ),
      transforms.CenterCrop(image_size),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
    ])

  def __len__(self) -> int:
    return len(self.image_paths)

  def __getitem__(self, index: int) -> torch.Tensor:
    image_path = self.image_paths[index]

    with Image.open(image_path) as image:
      image = image.convert("RGB")
      return self.transform(image)
