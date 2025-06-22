# Autoencoders_for_image_colorization

An image colorization project using convolutional autoencoders to transform grayscale images into realistic colored versions. The model is trained on the Microsoft COCO 2017 dataset and implements a U-Net architecture for high-quality image-to-image translation.

## Project Overview

This project demonstrates the use of deep learning for automatic image colorization. By training a convolutional autoencoder on the MS COCO dataset, the model learns to predict color information (a, b channels in Lab color space) from grayscale input (L channel).

## Architecture

This model uses a convolutional autoencoder with U-Net autoencoder with skip connections. Input is 256x256 grayscale, output is the same size with color predictions.

- **Encoder**: Conv2D layers that downsample
- **Decoder**: UpSampling layers that go back to original size
- **Key Components**: Skip connections between encoder and decoder blocks

## Project Structure

```
Autoencoders_for_image_colorization/
│
├── data/
│   ├── color_bins/
│   │   └── pts_in_hull.npy         # Color bins for classification
│   ├── Grayscaleimage.png
│   └── Ground_truth.png
│
├── models/
│   ├── mse/
│   │   ├── best_model.keras        # Best model using regression loss
│   │   └── final_model.keras
│   └── sparse_crossentropy/
│       └── best_model.keras        # Best model using classification loss
│
├── notebooks/
│   └── exploratory_analysis.ipynb  # Data exploration and analysis
│
├── outputs/
│   ├── mse/
│   │   ├── colorized_img.jpg       # Output using regression loss
│   │   └── training_hist.png       # Training history
│   └── sparse_crossentropy/
│       ├── colorized_img.jpg       # Output using classification loss
│       └── training_hist.png       # Trainng history
│
│
├── src/
│   ├── __init__.py
│   ├── config.py                   # Configuration parameters
│   ├── data_loader.py              # Data loading and preprocessing
│   ├── model.py                    # Model architecture
│   ├── train.py                    # Training script
│   ├── evaluate.py                 # Model evaluation
│   └── colorize.py                 # Script to colorize
│
├── .gitignore
├── README.md
└── requirements.txt                # Python dependencies
```

## Technical Specifications

### Requirements

- **GPU**: NVIDIA RTX 3050Ti
- **RAM**: 8GB+ recommended
- **Python**: 3.10
- **CUDA Toolkit**: 11.2
- **cuDNN**: 8.1.0
- **NumPy**: 1.23.5
- **TensorFlow**: 2.10.0

## Dataset Information

### MS COCO 2017 Dataset

- **Training Set**: 60,000 images
- **Validation Set**: ~5,000 images
- **Test Set**: ~40,670 images
- **Image Res**: 256×256 pixel

## Installation

```bash
git clone https://github.com/stashorph/Autoencoders_for_image_colorization.git
cd Autoencoders_for_image_colorization
python -m venv colorization_env
colorization_env\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Usage

### Training the Model

```bash
python src/train.py
```

### Colorizing Images

```bash
python src/colorize.py --input path/to/grayscale/image.jpg --output path/to/colorized/output.jpg
```

### Evaluating the Model

```bash
python src/evaluate.py --model models/best_model.keras
```

### Data Analysis

```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

## Evaluation Metrics

- **Mean Squared Error (MSE)**: Primary loss function for training
- **Peak Signal-to-Noise Ratio (PSNR)**: Reconstruction quality metric
- **Structural Similarity Index Measure (SSIM)**: Structural similarity metric

## Results

### Sample Outputs

Check the `outputs/` directory for sample colorization results including:

- Original grayscale inputs
- Ground truth colored images
- Model predictions
- Training history visualizations

## Configuration

Key parameters can be modified in `src/config.py`:

- Dataset sample sizes
- Training hyperparameters
- Model architecture
- File paths and directories

## Limitations

- **Old Dependencies**: Locked to TensorFlow 2.10 and Python 3.10 for Windows GPU compatibility
- **Hardware constraints**: Performance limited by laptop grade GPU
- **Dull outputs**: MSE loss, results in muddy and dull outputs. Due to averaging color predictions.
