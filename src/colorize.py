import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

import config
from model import ColorizationMetrics

def preprocess_grayscale(gs_path):

    img_bgr = cv2.imread(gs_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Grayscale image not found at: {gs_path}")
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (256, 256))
    

    img_rgb_float = img_rgb.astype(np.float32) / 255.0
    lab_image = cv2.cvtColor(img_rgb_float, cv2.COLOR_RGB2Lab)
    
    # Normalize the L channel for model
    L = lab_image[:, :, 0]
    L_norm = (L / 50.0) - 1.0
    L = L_norm[np.newaxis, :, :, np.newaxis]
    
    return L

def deprocess_lab_to_rgb(L, ab):

    # Denormalize the L and ab channels
    L = (L + 1.0) * 50.0
    ab = ab * 128.0
    lab_image = np.concatenate((L, ab), axis=-1)
    
    rgb_image = cv2.cvtColor(lab_image.astype(np.float32), cv2.COLOR_Lab2RGB)
    rgb_image = np.clip(rgb_image, 0, 1)
    
    return rgb_image

def colorize():

    gs_path = os.path.join(config.DATA_DIR, "grayscale.png")
    gt_path = os.path.join(config.DATA_DIR, "Ground_truth.jpg")
    output_path = os.path.join(config.OUTPUTS_DIR, "output.png")

    L = preprocess_grayscale(gs_path)

    model = tf.keras.models.load_model(config.BEST_MODEL_PATH, custom_objects={
        "psnr": ColorizationMetrics.psnr,
        "ssim": ColorizationMetrics.ssim,
        "color_accuracy": ColorizationMetrics.color_accuracy,
    })

    predicted_ab = model.predict(L)[0] # Remove batch dimension

    colorized_img = deprocess_lab_to_rgb(L[0], predicted_ab)
    grayscale_img = cv2.resize(cv2.cvtColor(cv2.imread(gs_path), cv2.COLOR_BGR2RGB), (256, 256))
    ground_truth = cv2.resize(cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB), (256, 256))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(grayscale_img)
    axes[0].set_title("Grayscale Input")

    axes[1].imshow(colorized_img)
    axes[1].set_title("Colorized Output")

    axes[2].imshow(ground_truth)
    axes[2].set_title("Ground Truth")

    fig.suptitle("Image Colorization Results")
    plt.tight_layout()
    
    plt.savefig(output_path)


if __name__ == '__main__':
    colorize()
