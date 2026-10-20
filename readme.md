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

### 2. Download and prepare the dataset

```bash
python -m scripts.setup
```

This downloads the Open Images ID list, samples non-overlapping train/test splits, downloads both splits and pre-resizes them to `cfg.image.image_size` into `data/open-images/{train,test}_images_<size>`. Re-running it skips images that are already downloaded or resized.

Defaults are at the top of [scripts/setup.py](scripts/setup.py) and can be overridden with flags:

| Flag | Default | Description |
| --- | --- | --- |
| `--train-count` | 400,000 | training images |
| `--test-count` | 500 | test images |
| `--seed` | 55 | split sampling seed |
| `--num-processes` | 35 | parallel downloads |
| `--image-size` | `cfg.image.image_size` | pre-resize size |
| `--cpu-fraction` | 0.8 | share of CPU cores used for pre-resizing |

```bash
python -m scripts.setup --train-count 50000 --num-processes 16
```

To only redo the resize (e.g. after changing `cfg.image.image_size`):

```bash
python -m scripts.preresize
```

### 3. Start training

```bash
python -m training.train_vae
```

### 4. Test recon (optional)
```bash
python -m tests.recon_test
```