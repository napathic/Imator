# Imator

A text-to-image and image-to-image model in development, for generating and editing images with text prompts.

**Currently working on:** VAE training.

## Train the VAE

Run all commands from the project root.

### 1. Set up Python

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download metadata → extract IDs → delete metadata

```bash
mkdir -p data/open-images

curl -fL \
  https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv \
  -o data/open-images/train_metadata.csv && \
awk -F, 'NR > 1 {
  gsub(/\r/, "", $1)
  print "train/" $1
}' data/open-images/train_metadata.csv \
  > data/open-images/dataset_ids.txt && \
rm -f data/open-images/train_metadata.csv
```

### 3. Create training and test splits

Default: **50,000 training images** and **500 test images**. Change `TRAIN_COUNT` and `TEST_COUNT` below to choose your own sizes.

The splits have no overlap; both are sampled from the Open Images training pool.

```bash
python - <<'PY'
import random
from pathlib import Path

TRAIN_COUNT = 250_000
TEST_COUNT = 500
SEED = 55

root = Path("data/open-images")
ids = list(dict.fromkeys(root.joinpath("dataset_ids.txt").read_text().splitlines()))
sample = random.Random(SEED).sample(ids, TRAIN_COUNT + TEST_COUNT)

for name, split in (
    ("train_ids.txt", sample[:TRAIN_COUNT]),
    ("test_ids.txt", sample[TRAIN_COUNT:]),
):
    destination = root / name
    destination.write_text("\n".join(split) + "\n")
    print(f"Wrote {len(split):,} IDs to {destination}")
PY
```

### 4. Download both splits

```bash
mkdir -p data/open-images/train_images data/open-images/test_images

python downloader.py \
  data/open-images/train_ids.txt \
  --download_folder=data/open-images/train_images \
  --num_processes=35

python downloader.py \
  data/open-images/test_ids.txt \
  --download_folder=data/open-images/test_images \
  --num_processes=35
```

### 5. Start training

```bash
python -m training.train_vae
```
