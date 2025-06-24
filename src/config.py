import os

# ** Project Paths **
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input/Output Directories
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# Model and Plot File Paths
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.keras")
FINAL_MODEL_PATH = os.path.join(MODELS_DIR, "final_model.keras")
TRAINING_PLOT_PATH = os.path.join(OUTPUTS_DIR, "training_history.png")
TEST_VIS_PATH = os.path.join(OUTPUTS_DIR, "test_predictions.png")


# ** Dataset Parameters **
DATASET_NAME = "coco-2017"
TRAIN_SPLIT = "train"
VAL_SPLIT = "validation"
TEST_SPLIT = "test"

# Number of samples to use from the dataset. (Set to None to use the entire dataset split.)
TRAIN_SAMPLES = 60000
VAL_SAMPLES = None
TEST_SAMPLES = 40664
RANDOM_SEED = 51


# ** Image Processing Parameters **
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)

INPUT_CHANNELS = 1
OUTPUT_CHANNELS = 2


# ** Model Hyperparameters **
LEARNING_RATE = 0.0001
LOSS_FUNCTION = "mse"
LEAKY_RELU_ALPHA = 0.2
OPTIMIZER_CLIPNORM = 1.0
COLOR_ACCURACY_THRESHOLD = 0.15


# ** Training Parameters **
EPOCHS = 100
BATCH_SIZE = 4
EARLY_STOPPING_PATIENCE = 5
LR_REDUCE_PATIENCE = 3
LR_REDUCE_FACTOR = 0.5
