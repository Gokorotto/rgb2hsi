import numpy as np
import matplotlib.pyplot as plt
from train_model import unet1, X_test, Y_test

# Predict on test set
predicted_hyper = unet1.predict(X_test)

def show_results(idx):
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))

    axes[0].imshow(X_test[idx])  # RGB image
    axes[0].set_title("RGB Image")

    axes[1].imshow(Y_test[idx][:, :, 1])  # Ground truth spectral band (example: band 20)
    axes[1].set_title("True Spectral Band")

    axes[2].imshow(predicted_hyper[idx][:, :, 1])  # Predicted spectral band
    axes[2].set_title("Predicted Spectral Band")

    plt.show()

# Show some test images
show_results(0)
show_results(1)