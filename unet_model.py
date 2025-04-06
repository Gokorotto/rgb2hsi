import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model

def unet_model(input_shape=(512, 512, 3), output_channels=5):
    inputs = Input(shape=input_shape)

    # Encoder (Contracting Path)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    # Bottleneck (Bottleneck layer)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)

    # Decoder (Expanding Path)
    u1 = UpSampling2D((2, 2))(c4)
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(u1)
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(c5)
    c5 = concatenate([c5, c3], axis=-1)

    u2 = UpSampling2D((2, 2))(c5)
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(u2)
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(c6)
    c6 = concatenate([c6, c2], axis=-1)

    u3 = UpSampling2D((2, 2))(c6)
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(u3)
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(c7)
    c7 = concatenate([c7, c1], axis=-1)

    # Output layer (using sigmoid activation for multi-channel output)
    outputs = Conv2D(output_channels, (1, 1), activation='sigmoid')(c7)

    model = Model(inputs, outputs)

    def ssim_metric(y_true, y_pred):
        return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    def euclidean_distance(y_true, y_pred):
        return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), axis=[1, 2, 3])))
    def psnr_metric(y_true, y_pred):
        return tf.image.psnr(y_true, y_pred, max_val=1.0)

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[ssim_metric, euclidean_distance, psnr_metric])

    return model