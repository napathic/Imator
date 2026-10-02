Imator would be a modern optimized simple to use text to Image model + image -> image features for photo editing based on a image and text


---------------------------------------------
currently working on:
Vae training.


---------------------------------------------
how to train the vae:

1. make and activate venv
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

2. first download the required metadata
curl -L \
  https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv \
  -o data/open-images/train_metadata.csv

3. convert it to a txt
awk -F, 'NR > 1 {
  gsub(/\r/, "", $1)
  print "train/" $1
}' data/open-images/train_metadata.csv \
  > data/open-images/train_ids.txt

4. (optional) delete the metadata
rm -f data/open-images/train_metadata.csv

5. Create a new file with random image ids from the set,
python - <<'PY'
import random
from pathlib import Path

source = Path("data/open-images/train_ids.txt")
destination = Path("data/open-images/test_ids.txt")

ids = source.read_text().splitlines()
random.seed(51)
sample = random.sample(ids, 5_000)

destination.write_text("\n".join(sample) + "\n")
print(f"Wrote {len(sample)} IDs to {destination}")
PY

6. start google downloader to download all thoes images
python downloader.py \
  data/open-images/test_ids.txt \
  --download_folder=data/open-images/images \
  --num_processes=5

7. start training the VAE
python -m training.train_vae