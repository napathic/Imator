from multiprocessing import Pool
from pathlib import Path
import os

from PIL import Image
from tqdm import tqdm

from config import cfg


IMAGE_ROOT = Path("data") / "open-images"
SPLITS = ("train_images", "test_images")
SUFFIXES = {".jpg", ".jpeg", ".png"}


def resized_dir(split: str, size: int) -> Path:
  return IMAGE_ROOT / f"{split}_{size}"


def resize_one(job: tuple[Path, Path, int]) -> bool:
  src, dst, size = job
  if dst.exists():
    return True

  try:
    with Image.open(src) as image:
      image.draft("RGB", (size * 2, size * 2))
      image = image.convert("RGB")

      w, h = image.size
      scale = size / min(w, h)
      new_w, new_h = max(size, round(w * scale)), max(size, round(h * scale))
      image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

      left = int(round((new_w - size) / 2))
      top = int(round((new_h - size) / 2))
      image = image.crop((left, top, left + size, top + size))

      tmp = dst.with_suffix(".tmp.png")
      image.save(tmp, icc_profile=None, exif=b"")
      tmp.replace(dst)
    return True
  except Exception:
    return False


def preresize(size: int, workers: int | None = None) -> None:
  for split in SPLITS:
    src_dir = IMAGE_ROOT / split
    if not src_dir.exists():
      print(f"Skipping {src_dir}, not found")
      continue

    dst_dir = resized_dir(split, size)
    dst_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
      (path, dst_dir / f"{path.stem}.png", size)
      for path in src_dir.iterdir()
        if path.suffix.lower() in SUFFIXES
    ]

    failed = 0
    with Pool(workers or os.cpu_count()) as pool:
      for ok in tqdm(pool.imap_unordered(resize_one, jobs, chunksize=64), total=len(jobs), desc=f"{split} -> {size}px"):
        failed += not ok

    print(f"Wrote {len(jobs) - failed:,} images to {dst_dir}" + (f" ({failed:,} failed)" if failed else ""))


def main() -> None:
  preresize(cfg.image.image_size)

if __name__ == "__main__":
  main()
