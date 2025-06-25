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

def build_autoencoder(input_channels, leaky_relu_alpha, height=256, width=256, num_bins=313):

    # Encoder
    inputs = Input(shape=(height, width, input_channels))

    # 256x256 -> 128x128
    e1 = Conv2D(64, (3, 3), strides=1, padding='same')(inputs)
    e1 = LeakyReLU(alpha=leaky_relu_alpha)(e1)
    e1_pool = Conv2D(64, (3, 3), strides=2, padding='same')(e1)
    e1_pool = BatchNormalization()(e1_pool)

    # 128x128 -> 64x64
    e2 = Conv2D(128, (3, 3), strides=1, padding='same')(e1_pool)
    e2 = LeakyReLU(alpha=leaky_relu_alpha)(e2)
    e2_pool = Conv2D(128, (3, 3), strides=2, padding='same')(e2)
    e2_pool = BatchNormalization()(e2_pool)

    # 64x64 -> 32x32
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
    # 32x32 -> 64x64
    d1 = UpSampling2D(size=(2, 2))(b)
    d1 = Concatenate()([d1, e3]) # Skip connection
    d1 = Conv2D(256, (3, 3), strides=1, padding='same')(d1)
    d1 = ReLU()(d1)
    d1 = BatchNormalization()(d1)

    # 64x64 -> 128x128
    d2 = UpSampling2D(size=(2, 2))(d1)
    d2 = Concatenate()([d2, e2])
    d2 = Conv2D(128, (3, 3), strides=1, padding='same')(d2)
    d2 = ReLU()(d2)
    d2 = BatchNormalization()(d2)

    # 128x128 -> 256x256
    d3 = UpSampling2D(size=(2, 2))(d2)
    d3 = Concatenate()([d3, e1])
    d3 = Conv2D(64, (3, 3), strides=1, padding='same')(d3)
    d3 = ReLU()(d3)
    d3 = BatchNormalization()(d3)

    # Output Layer
    outputs = Conv2D(num_bins, (1, 1), activation='softmax', padding='same')(d3)

    model = Model(inputs=inputs, outputs=outputs, name="unet_colorization_model")
    return model

def get_compiled_model():

    model = build_autoencoder(
        input_channels=config.INPUT_CHANNELS,
        leaky_relu_alpha=config.LEAKY_RELU_ALPHA
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.LEARNING_RATE,
        clipnorm=config.OPTIMIZER_CLIPNORM
    )

    model.compile(
        optimizer=optimizer,
        loss=config.LOSS_FUNCTION,
        metrics= ['sparse_categorical_accuracy']
    )
    return model

