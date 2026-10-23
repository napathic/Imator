# python3
# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Open Images image downloader.

This script downloads a subset of Open Images images, given a list of image ids.
Typical uses of this tool might be downloading images:
- That contain a certain category.
- That have been annotated with certain types of annotations (e.g. Localized
Narratives, Exhaustively annotated people, etc.)

The input file IMAGE_LIST should be a text file containing one image per line
with the format <SPLIT>/<IMAGE_ID>, where <SPLIT> is either "train", "test",
"validation", or "challenge2018"; and <IMAGE_ID> is the image ID that uniquely
identifies the image in Open Images. A sample file could be:
  train/f9e0434389a1d4dd
  train/1a007563ebc18664
  test/ea8bfd4e765304db

"""

import argparse
from concurrent import futures
from pathlib import Path
import os
import re
import sys
import threading

import tqdm
import urllib3

from scripts.preresize import resize_image

BASE_URL = 'https://open-images-dataset.s3.amazonaws.com'
REGEX = r'(test|train|validation|challenge2018)/([a-fA-F0-9]*)'


def check_and_homogenize_one_image(image):
  split, image_id = re.match(REGEX, image).groups()
  yield split, image_id


def check_and_homogenize_image_list(image_list):
  for line_number, image in enumerate(image_list):
    try:
      yield from check_and_homogenize_one_image(image)
    except (ValueError, AttributeError):
      raise ValueError(
          f'ERROR in line {line_number} of the image list. The following image '
          f'string is not recognized: "{image}".')


def read_image_list_file(image_list_file):
  with open(image_list_file, 'r') as f:
    for line in f:
      yield line.strip().replace('.jpg', '')


def finish_resize(future, split, image_id, resize_slots, on_resized):
  resize_slots.release()
  ok = not future.cancelled() and future.exception() is None and future.result()
  if not ok:
    tqdm.tqdm.write(f'Skipping `{split}/{image_id}`: resize failed')
  on_resized()


def download_one_image(http, split, image_id, download_folder, image_size=None,
                       resize_pool=None, resize_slots=None, on_resized=None):
  extension = 'png' if image_size else 'jpg'
  destination = os.path.join(download_folder, f'{image_id}.{extension}')
  if os.path.isfile(destination) and os.path.getsize(destination) > 0:
    return 'existing'

  try:
    response = http.request('GET', f'{BASE_URL}/{split}/{image_id}.jpg')
  except urllib3.exceptions.HTTPError as exception:
    tqdm.tqdm.write(f'Skipping `{split}/{image_id}`: {exception}')
    return 'failed'

  if response.status != 200:
    tqdm.tqdm.write(f'Skipping `{split}/{image_id}`: HTTP {response.status}')
    return 'failed'

  body = response.data
  if image_size:
    resize_slots.acquire()
    try:
      future = resize_pool.submit(resize_image, body, Path(destination), image_size)
    except Exception:
      resize_slots.release()
      raise
    future.add_done_callback(
        lambda f: finish_resize(f, split, image_id, resize_slots, on_resized))
    return 'resizing'

  temporary = destination + '.part'
  with open(temporary, 'wb') as f:
    f.write(body)
  os.replace(temporary, destination)
  return 'downloaded'


def download_all_images(args):
  """Downloads all images specified in the input file."""
  http = urllib3.PoolManager(
      maxsize=args['num_processes'],
      timeout=urllib3.Timeout(connect=10, read=60),
      retries=urllib3.Retry(
          total=10, backoff_factor=0.5,
          status_forcelist=(429, 500, 502, 503, 504)))

  download_folder = args['download_folder'] or os.getcwd()

  if not os.path.exists(download_folder):
    os.makedirs(download_folder)

  try:
    image_list = list(
        check_and_homogenize_image_list(
            read_image_list_file(args['image_list'])))
  except ValueError as exception:
    sys.exit(exception)

  image_size = args.get('image_size')
  resize_workers = args.get('resize_workers') or os.cpu_count()
  resize_pool = futures.ProcessPoolExecutor(
      max_workers=resize_workers) if image_size else None
  resize_slots = threading.BoundedSemaphore(resize_workers * 4)

  progress_bar = tqdm.tqdm(
      total=len(image_list), desc='Downloading images', leave=True)
  progress_lock = threading.Lock()

  def advance():
    with progress_lock:
      progress_bar.update(1)

  try:
    with futures.ThreadPoolExecutor(
        max_workers=args['num_processes']) as executor:
      all_futures = [
          executor.submit(download_one_image, http, split, image_id,
                          download_folder, image_size, resize_pool,
                          resize_slots, advance)
          for (split, image_id) in image_list
      ]
      for future in futures.as_completed(all_futures):
        if future.result() != 'resizing':
          advance()
  finally:
    if resize_pool:
      resize_pool.shutdown()
    progress_bar.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument(
      'image_list',
      type=str,
      default=None,
      help=('Filename that contains the split + image IDs of the images to '
            'download. Check the document'))
  parser.add_argument(
      '--num_processes',
      type=int,
      default=15,
      help='Number of parallel processes to use (default is 5).')
  parser.add_argument(
      '--download_folder',
      type=str,
      default=None,
      help='Folder where to download the images.')
  download_all_images(vars(parser.parse_args()))
