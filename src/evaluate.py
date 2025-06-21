
import config
from data_loader import get_dataset
from model import ColorizationMetrics

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

def deprocess_lab_to_rgb(L, ab):
    L = (L + 1.0) * 50.0
    ab = ab * 128.0
    
    # Combine the L and ab channels to form a full Lab image
    lab = np.concatenate((L, ab), axis=-1)
    rgb = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_Lab2RGB)
    rgb = np.clip(rgb, 0, 1)
    
    return rgb

def run_evaluation(model_path):

    print(f"Loading model from: {model_path}")


    model = tf.keras.models.load_model(model_path, custom_objects={
        "psnr": ColorizationMetrics.psnr,
        "ssim": ColorizationMetrics.ssim,
        "color_accuracy": ColorizationMetrics.color_accuracy,
    })

    test_dataset = get_dataset('test')

    results = model.evaluate(test_dataset, verbose=1)

    print("\n Evaluation Results")
    for name, val in zip(model.metrics_names, results):
        print(f"{name}: {val:.3f}")


    for gray, true_color in test_dataset.take(1):
        ab_pred_batch = model.predict(gray)
        
        fig, axes = plt.subplots(5, 3, figsize=(12, 20))
        fig.suptitle("Model Predictions vs. Ground Truth")

        for i in range(5):
            grayscale_img = deprocess_lab_to_rgb(gray[i], np.zeros_like(ab_pred_batch[i]))
            predicted_img = deprocess_lab_to_rgb(gray[i], ab_pred_batch[i])
            gt_color = deprocess_lab_to_rgb(gray[i], true_color[i])

            axes[i, 0].imshow(grayscale_img)
            axes[i, 0].set_title("Grayscale Image")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(predicted_img)
            axes[i, 1].set_title("Predicted Image")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(gt_color)
            axes[i, 2].set_title("Ground Truth")
            axes[i, 2].axis("off")
        
        plt.tight_layout()
        plt.savefig(config.TEST_VIS_PATH)
        break

if __name__ == '__main__':
    run_evaluation(config.BEST_MODEL_PATH)
