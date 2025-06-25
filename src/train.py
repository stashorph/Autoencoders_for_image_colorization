import tensorflow as tf
from tensorflow.keras.callbacks import ( #type: ignore
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)
import matplotlib.pyplot as plt

import config
from data_loader import get_dataset
from model import get_compiled_model

def run_training():

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


    train_dataset = get_dataset(split="train")
    val_dataset = get_dataset(split="validation")
    print("Datasets loaded.\n")

    # Build and compile the model
    model = get_compiled_model()

    # Callbacks
    model_checkpoint = ModelCheckpoint(
        filepath=config.BEST_MODEL_PATH,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=config.EARLY_STOPPING_PATIENCE,
        verbose=1,
        restore_best_weights=True
    )

    lr_reduce = ReduceLROnPlateau(
        monitor='val_loss',
        factor=config.LR_REDUCE_FACTOR,
        patience=config.LR_REDUCE_PATIENCE,
        verbose=1,
        mode='min'
    )

    # Training
    history = model.fit(
        train_dataset,
        epochs=config.EPOCHS,
        validation_data=val_dataset,
        callbacks= [
        model_checkpoint,
        early_stop,
        lr_reduce
    ]
    )
    print("\n Training Finished ")

    # Save the final model and plot
    model.save(config.FINAL_MODEL_PATH)
    
    plot_training_history(history)
    

def plot_training_history(history):
    
    plt.style.use("fivethirtyeight")
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend(), plt.title('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['sparse_categorical_accuracy'], label='Train Acc')
    plt.plot(history.history['val_sparse_categorical_accuracy'], label='Val Acc')
    plt.legend(), plt.title('Accuracy')

    plt.tight_layout()
    plt.savefig(config.TRAINING_PLOT_PATH)

if __name__ == '__main__':
    run_training()
