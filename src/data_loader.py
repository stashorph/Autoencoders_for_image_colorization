import tensorflow as tf
from tqdm import tqdm
import fiftyone.zoo as foz
import cv2
import numpy as np

# Import settings from the config.py file
import config

def load_fiftyone_dataset(split, max_samples, shuffle, seed):
    
    # Load the dataset from the zoo
    dataset = foz.load_zoo_dataset(
        config.DATASET_NAME,
        split=split,
        max_samples=max_samples,
        shuffle=shuffle,
        seed=seed
    )
    dataset.persistent = True
    return dataset

def is_image_valid(filepath):

    # Returns True if the image is valid, False otherwise.
    try:
        img_bytes = tf.io.read_file(filepath)
        tf.image.decode_jpeg(img_bytes, channels=3)
        return True
    except Exception:
        return False

def process_path(filepath):
    # Reads, resizes, and preprocesses the image.

    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [config.IMAGE_HEIGHT, config.IMAGE_WIDTH])
    img.set_shape([config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 3])
    
    # Normalize the RGB image to the [0, 1] range for color space conversion
    img = tf.cast(img, tf.float32) / 255.0

    # Convert RGB to Lab color space using OpenCV
    def to_lab(rgb_img):
        lab_img = cv2.cvtColor(rgb_img.numpy(), cv2.COLOR_RGB2Lab)
        return lab_img.astype(np.float32)

    lab_image = tf.py_function(func=to_lab, inp=[img], Tout=tf.float32)
    lab_image.set_shape([config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 3])

    # Separate the L channel (input) and ab channels (target)
    L_channel = lab_image[..., 0:1]
    ab_channels = lab_image[..., 1:]

    L_channel_norm = (L_channel / 50.0) - 1.0
    ab_channels_norm = ab_channels / 128.0

    return L_channel_norm, ab_channels_norm

def get_dataset(split):
    
    # Creates a tf.Dataset for a given split (train, val, or test).
    is_training = split == config.TRAIN_SPLIT

    if split == config.TRAIN_SPLIT:
        max_samples = config.TRAIN_SAMPLES
    elif split == config.VAL_SPLIT:
        max_samples = config.VAL_SAMPLES
    else:
        max_samples = config.TEST_SAMPLES

    fo_dataset = load_fiftyone_dataset(
        split=split,
        max_samples=max_samples,
        shuffle=is_training,
        seed=config.RANDOM_SEED
    )
    
    filepaths = fo_dataset.values("filepath")

    valid_filepaths = [
        f for f in tqdm(filepaths, desc=f"Filtering {split} images") 
        if is_image_valid(f)
    ]

    dataset = tf.data.Dataset.from_tensor_slices(valid_filepaths)

    if is_training:
        dataset = dataset.shuffle(buffer_size=1000)
    
    dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch the dataset
    dataset = dataset.batch(config.BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset


