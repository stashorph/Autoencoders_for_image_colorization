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
│   ├── Grayscaleimage.png          # Sample grayscale input
│   └── Ground_truth.png            # Sample ground truth colored image
│
├── models/
│   ├── best_model.keras            # Best model saved during training
│   └── final_model.keras           # Final trained model
│
├── notebooks/
│   └── exploratory_analysis.ipynb  # Data exploration and analysis
│
├── outputs/
│   ├── colorized_output.jpg        # Colorization result
│   ├── output.png                  # Additional sample outputs
│   └── continued_training.png      # Training history visualization
│
├── src/
│   ├── __init__.py
│   ├── config.py                   # Configuration parameters
│   ├── data_loader.py              # Data loading and preprocessing
│   ├── model.py                    # Model architecture
│   ├── train.py                    # Training script
│   ├── evaluate.py                 # Model evaluation
│   ├── colorize.py                 # Script for colorization
│   └── verify_gpu.py               # Gpu Verification
│
├── .gitignore
├── README.md
└── requirements.txt                # Python dependencies
```

## Technical Specifications

### Requirements

- **GPU**: NVIDIA GeForce RTX 3050 Ti Laptop GPU (or compatible)
- **RAM**: 8GB+ recommended for dataset loading
- **Python**: 3.10
- **CUDA Toolkit**: 11.2
- **cuDNN**: 8.1.0
- **NumPy**: 1.23.5
- **TensorFlow**: 2.10.0

## Dataset Information

### MS COCO 2017 Dataset

- **Training Set**: 90,000 images
- **Validation Set**: ~5,000 images (full validation split)
- **Test Set**: ~40,670 images (full test split)
- **Image Resolution**: 256×256 pixels (preprocessed)

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

### Jupyter Notebook Analysis

```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

## Evaluation Metrics

- **Mean Squared Error (MSE)**: Primary loss function for training
- **Peak Signal-to-Noise Ratio (PSNR)**: Reconstruction quality metric
- **Structural Similarity Index Measure (SSIM)**: Perceptual similarity assessment

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

- **Legacy Dependencies**: Locked to TensorFlow 2.10 and Python 3.10 for Windows GPU compatibility
- **Hardware Constraints**: Performance limited by laptop grade GPU
- **Dataset Size**: Large dataset requires significant storage and processing time
