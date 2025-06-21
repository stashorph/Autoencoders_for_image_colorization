import config
from data_loader import get_dataset
from model import get_compiled_model
import tensorflow as tf
from tensorflow.keras.callbacks import ( #type: ignore
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)
import matplotlib.pyplot as plt

def run_training():

    print("Starting Autoencoder Model Training")

    gpu = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpu, True)

    train_dataset = get_dataset('train')
    val_dataset = get_dataset('validation')
    print("Datasets loaded.\n")

    # Build and compile the model
    model = get_compiled_model()

    # Callbacks
    model_checkpoint = ModelCheckpoint(
        filepath=config.BEST_MODEL_PATH,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
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

    callbacks = [
        model_checkpoint,
        early_stop,
        lr_reduce
    ]

    # Training
    print(f"\nStarting training for {config.EPOCHS} epochs with batch size {config.BATCH_SIZE}")
    history = model.fit(
        train_dataset,
        epochs=config.EPOCHS,
        validation_data=val_dataset,
        callbacks=callbacks
    )
    print("\n Training Finished ")

    # Save the final model and plot
    print(f"Saving final model to '{config.FINAL_MODEL_PATH}'...")
    model.save(config.FINAL_MODEL_PATH)
    
    plot(history)
    

def plot(history):
    
    plt.style.use("fivethirtyeight ")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # Plot training/validation loss
    ax1.plot(history.history['loss'], label='Training loss', color='blue')
    ax1.plot(history.history['val_loss'], label='Val loss', color='orange')
    ax1.set_title('Loss vs. Epochs')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss (MSE)')
    ax1.legend()

    # Plot mae
    ax2.plot(history.history['mae'], label='Train MAE', color='green')
    ax2.plot(history.history['val_mae'], label='Val MAE', color='red')
    ax2.set_title('MAE vs. Epochs')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('MAE')
    ax2.legend()
    
    fig.suptitle('Model Training History')
    plt.savefig(config.TRAINING_PLOT_PATH)

if __name__ == '__main__':
    run_training()
