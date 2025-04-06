import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.losses import Huber
from unet_model import unet_model

# Load preprocessed images
X_train = np.load('rgb_images.npy')
Y_train = np.load('hyper_images.npy')

# Normalize the images
X_train = X_train / 255.0
Y_train = (Y_train - Y_train.min()) / (Y_train.max() - Y_train.min())

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

# Initialize the model
unet1 = unet_model()

# Print model summary
unet1.summary()

# Callbacks: ReduceLROnPlateau and EarlyStopping
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
]

EPOCHS = 100
BATCH_SIZE = 4
history1 = unet1.fit(
    X_train, Y_train,
    validation_data=(X_test, Y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)