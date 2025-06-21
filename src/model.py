import tensorflow as tf
from tensorflow.keras.layers import ( #type: ignore
    Input,
    Conv2D,
    UpSampling2D,
    Concatenate,
    BatchNormalization,
    LeakyReLU,
    ReLU
) 
from tensorflow.keras.models import Model #type: ignore

# Import from config.py file
import config

class ColorizationMetrics:
    @staticmethod
    def psnr(y_true, y_pred):
        return tf.image.psnr(y_true, y_pred, max_val=2.0) # Max_val is set to 2 as normalized range is [-1, 1]

    @staticmethod
    def ssim(y_true, y_pred):
        return tf.image.ssim(y_true, y_pred, max_val=2.0)

    @staticmethod
    def color_accuracy(y_true, y_pred):
        threshold = config.COLOR_ACCURACY_THRESHOLD
        # Check if difference between true and predicted values are within threshold
        delta = tf.abs(y_true - y_pred)
        correct_pixels = tf.cast(delta < threshold, tf.float32)

        return tf.reduce_mean(correct_pixels)

def build_autoencoder(height, width, input_channels, output_channels, leaky_relu_alpha):

    # Encoder
    inputs = Input(shape=(height, width, input_channels), name="input_L_channel")

    # Block 1: 256x256 -> 128x128
    e1 = Conv2D(64, (3, 3), strides=1, padding='same')(inputs)
    e1 = LeakyReLU(alpha=leaky_relu_alpha)(e1)
    e1_pool = Conv2D(64, (3, 3), strides=2, padding='same')(e1)
    e1_pool = BatchNormalization()(e1_pool)

    # Block 2: 128x128 -> 64x64
    e2 = Conv2D(128, (3, 3), strides=1, padding='same')(e1_pool)
    e2 = LeakyReLU(alpha=leaky_relu_alpha)(e2)
    e2_pool = Conv2D(128, (3, 3), strides=2, padding='same')(e2)
    e2_pool = BatchNormalization()(e2_pool)

    # Block 3: 64x64 -> 32x32
    e3 = Conv2D(256, (3, 3), strides=1, padding='same')(e2_pool)
    e3 = LeakyReLU(alpha=leaky_relu_alpha)(e3)
    e3_pool = Conv2D(256, (3, 3), strides=2, padding='same')(e3)
    e3_pool = BatchNormalization()(e3_pool)

    # Bottleneck
    # 32x32
    b = Conv2D(512, (3, 3), strides=1, padding='same')(e3_pool)
    b = LeakyReLU(alpha=leaky_relu_alpha)(b)
    b = BatchNormalization()(b)

    # Decoder
    # Block 1: 32x32 -> 64x64
    d1 = UpSampling2D(size=(2, 2))(b)
    d1 = Concatenate()([d1, e3]) # Skip connection
    d1 = Conv2D(256, (3, 3), strides=1, padding='same')(d1)
    d1 = ReLU()(d1)
    d1 = BatchNormalization()(d1)

    # Block 2: 64x64 -> 128x128
    d2 = UpSampling2D(size=(2, 2))(d1)
    d2 = Concatenate()([d2, e2])
    d2 = Conv2D(128, (3, 3), strides=1, padding='same')(d2)
    d2 = ReLU()(d2)
    d2 = BatchNormalization()(d2)

    # Block 3: 128x128 -> 256x256
    d3 = UpSampling2D(size=(2, 2))(d2)
    d3 = Concatenate()([d3, e1])
    d3 = Conv2D(64, (3, 3), strides=1, padding='same')(d3)
    d3 = ReLU()(d3)
    d3 = BatchNormalization()(d3)

    # Output Layer, 2-channel 'ab' output
    outputs = Conv2D(output_channels, (1, 1), activation='tanh', padding='same', name="output_ab_channels")(d3)

    model = Model(inputs=inputs, outputs=outputs, name="unet_colorization_model")
    return model

def get_compiled_model():

    # Build the model architecture
    model = build_autoencoder(
        height=config.IMAGE_HEIGHT,
        width=config.IMAGE_WIDTH,
        input_channels=config.INPUT_CHANNELS,
        output_channels=config.OUTPUT_CHANNELS,
        leaky_relu_alpha=config.LEAKY_RELU_ALPHA
    )

    # Define the optimizer
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.LEARNING_RATE,
        clipnorm=config.OPTIMIZER_CLIPNORM
    )

    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss=config.LOSS_FUNCTION,
        metrics=[
            'mae',
            ColorizationMetrics.psnr,
            ColorizationMetrics.ssim,
            ColorizationMetrics.color_accuracy
        ]
    )
    return model

