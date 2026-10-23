from pathlib import Path
from urllib.request import urlopen
import argparse
import os
import random
import csv

from tqdm import tqdm

from config import cfg
from downloader import download_all_images
from scripts.preresize import IMAGE_ROOT, resized_dir


METADATA_URL = "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv"
TRAIN_COUNT = 300_000
TEST_COUNT = 500
SEED = 55
NUM_PROCESSES = 100
IMAGE_SIZE = cfg.image.image_size
CPU_FRACTION = 0.9


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Download Open Images, split it and resize it on the fly.")
  parser.add_argument("--train-count", type=int, default=TRAIN_COUNT, help=f"training images (default {TRAIN_COUNT:,})")
  parser.add_argument("--test-count", type=int, default=TEST_COUNT, help=f"test images (default {TEST_COUNT:,})")
  parser.add_argument("--seed", type=int, default=SEED, help=f"split sampling seed (default {SEED})")
  parser.add_argument("--num-processes", type=int, default=NUM_PROCESSES, help=f"parallel downloads (default {NUM_PROCESSES})")
  parser.add_argument("--image-size", type=int, default=IMAGE_SIZE, help=f"pre-resize size (default cfg.image.image_size = {IMAGE_SIZE})")
  parser.add_argument("--cpu-fraction", type=float, default=CPU_FRACTION, help=f"share of cpu cores used for pre-resizing (default {CPU_FRACTION})")
  return parser.parse_args()


def fetch_ids(root: Path) -> Path:
  ids_file = root / "dataset_ids.txt"
  if ids_file.exists():
    print(f"Using existing {ids_file}")
    return ids_file

  metadata = root / "train_metadata.csv"
  with urlopen(METADATA_URL) as response, metadata.open("wb") as f:
    total = int(response.headers.get("Content-Length", 0)) or None
    with tqdm(total=total, unit="B", unit_scale=True, desc="Downloading metadata") as bar:
      while chunk := response.read(1 << 20):
        f.write(chunk)
        bar.update(len(chunk))

  with metadata.open(newline="") as f:
    rows = csv.reader(f)
    next(rows)
    ids_file.write_text("".join(f"train/{row[0].strip()}\n" for row in rows))

  metadata.unlink()
  print(f"Wrote {ids_file}")
  return ids_file


def make_splits(root: Path, ids_file: Path, train_count: int, test_count: int, seed: int) -> dict[str, Path]:
  ids = list(dict.fromkeys(ids_file.read_text().splitlines()))
  sample = random.Random(seed).sample(ids, train_count + test_count)

  splits = {
    "test": (root / "test_ids.txt", sample[train_count:]),
    "train": (root / "train_ids.txt", sample[:train_count]),
  }
  for destination, split in splits.values():
    destination.write_text("\n".join(split) + "\n")
    print(f"Wrote {len(split):,} IDs to {destination}")

  return {name: destination for name, (destination, _) in splits.items()}


def main() -> None:
  args = parse_args()
  root = IMAGE_ROOT
  root.mkdir(parents=True, exist_ok=True)

  ids_file = fetch_ids(root)
  split_files = make_splits(root, ids_file, args.train_count, args.test_count, args.seed)
  resize_workers = max(1, int((os.cpu_count() or 1) * args.cpu_fraction))

  for name, ids in split_files.items():
    download_all_images({
      "image_list": str(ids),
      "download_folder": str(resized_dir(f"{name}_images", args.image_size)),
      "num_processes": args.num_processes,
      "image_size": args.image_size,
      "resize_workers": resize_workers,
    })

  if args.image_size != cfg.image.image_size:
    print(f"Note: training uses cfg.image.image_size = {cfg.image.image_size}, set it to {args.image_size} to use these images")


if __name__ == "__main__":
  main()
