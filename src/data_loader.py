import config
import tensorflow as tf
from tqdm import tqdm
import fiftyone.zoo as foz
import cv2
import numpy as np
from scipy.spatial import cKDTree

color_bins = np.load(config.COLOR_BINS_PATH)
tree = cKDTree(color_bins)

def load_dataset(split, max_samples, shuffle, seed):
    
    # Load the dataset
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split=split,
        max_samples=max_samples,
        shuffle=shuffle,
        seed=seed
    )
    dataset.persistent = True
    return dataset

def is_image_valid(filepath):

    # Returns true if the image is valid, false otherwise.
    try:
        img_bytes = tf.io.read_file(filepath)
        tf.image.decode_jpeg(img_bytes, channels=3)
        return True
    except Exception:
        return False

def process_path(filepath):
    # Reads, resizes and preprocesses the image.

    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [256, 256])
    img.set_shape([256, 256, 3])
    
    # Normalize the RGB image to the [0, 1] range for color space conversion
    img = tf.cast(img, tf.float32) / 255.0

    def lab_and_class(rgb_img):

        # Convert RGB to Lab
        lab_img = cv2.cvtColor(rgb_img.numpy(), cv2.COLOR_RGB2Lab)
        L = lab_img[:, :, 0]
        ab = lab_img[:, :, 1:]

        h, w, _ = ab.shape
        ab_reshaped = ab.reshape((h * w, 2))
        
        # Find the nearest color bin for each pixel using KD-Tree
        _, indices = tree.query(ab_reshaped)
        class_indices = indices.reshape((h, w))
        
        return L.astype(np.float32), class_indices.astype(np.int32)


    L_channel, class_indices = tf.py_function(
        func=lab_and_class, inp=[img], Tout=[tf.float32, tf.int32]
    )

    # Normalize L channel
    L_norm = (L_channel / 50.0) - 1.0
    L_norm = tf.expand_dims(L_norm, axis=-1) # Add channel dim
    
    L_norm.set_shape([256, 256, 1])
    class_indices.set_shape([256, 256])
    
    return L_norm, class_indices


def get_dataset(split):
    
    # Creates a tf.Dataset for a given split (train, val, or test).
    is_training = split == "train"

    if split == "train":
        max_samples = config.TRAIN_SAMPLES
    elif split == "validation":
        max_samples = config.VAL_SAMPLES
    else:
        max_samples = config.TEST_SAMPLES

    fo_dataset = load_dataset(
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


