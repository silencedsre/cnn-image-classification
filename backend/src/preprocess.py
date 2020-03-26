import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import  Image
import os
import numpy as np
import pathlib

from src.config import TRAIN_PATH

BATCH_SIZE = 32
IMG_HEIGHT = 150
IMG_WIDTH = 150

AUTOTUNE = tf.data.experimental.AUTOTUNE


data_dir = pathlib.Path(TRAIN_PATH)
image_count = len(list(data_dir.glob('*/*.jpg')))
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)


CLASS_NAMES = np.array([item.name for item in data_dir.glob('*')])

list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'))

def get_label(file_path):
  # convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  class_labels = parts[-2] == CLASS_NAMES
  class_labels = tf.dtypes.cast(class_labels, tf.float32)
  return class_labels


def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])


def process_path(file_path):
  label = get_label(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label


# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

print(labeled_ds)

for image, label in labeled_ds.take(1):
  print("Image shape: ", image.numpy().shape)
  print("Label: ", label.numpy())


def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
  # This is a small dataset, only load it once, and keep it in memory.
  # use `.cache(filename)` to cache preprocessing work for datasets that don't
  # fit in memory.
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()

  ds = ds.shuffle(buffer_size=shuffle_buffer_size)

  # Repeat forever
  # ds = ds.repeat()

  ds = ds.batch(BATCH_SIZE)

  # `prefetch` lets the dataset fetch batches in the background while the model
  # is training.
  ds = ds.prefetch(buffer_size=AUTOTUNE)

  return ds


train_ds = prepare_for_training(labeled_ds)


# print(train_ds)
# image_batch, label_batch = next(iter(train_ds))
# print(label_batch)
