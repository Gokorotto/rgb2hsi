import os
import glob
import numpy as np
import cv2
#import spectral.io.envi as envi

# Paths
RGB_PATH = "rgb/*"
HYPER_PATH = "hyper/*"

# Get sorted file names (without extensions)
rgb_files = sorted(glob.glob(os.path.join(RGB_PATH, "*.jpg")))
hyper_files = sorted(glob.glob(os.path.join(HYPER_PATH, "*.hdr")))

# Extract base filenames without extensions
rgb_names = {os.path.splitext(os.path.basename(f))[0] for f in rgb_files}
hyper_names = {os.path.splitext(os.path.basename(f))[0] for f in hyper_files}

# Find matching files
common_names = sorted(rgb_names & hyper_names)  # Ensure only matching files are included

# Update file lists to only include matched pairs
rgb_files = [os.path.join(RGB_PATH, f"{name}.jpg") for name in common_names]
hyper_files = [os.path.join(HYPER_PATH, f"{name}.hdr") for name in common_names]

# Target spectral bands (zero-indexed)
TARGET_BANDS = [5, 10, 20, 30, 40]  # Example: 5th, 10th, 20th, 30th, and 40th bands

IMG_SIZE = (512, 512)  # New resolution

def load_rgb_image(img_path):
    """Load and preprocess an RGB image."""
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMG_SIZE)  # Resize to 512x512
    img = img.astype(np.float32) / 255.0  # Normalize
    return img

def load_hyperspectral_image(hyper_path):
    """Load hyperspectral image from RAW file and extract specific bands."""
    hyper_image = spectral.open_image(hyper_path)
    selected_bands = hyper_image[:, :, TARGET_BANDS]
    selected_bands_resized = np.array([cv2.resize(selected_bands[:, :, i], IMG_SIZE) for i in range(len(TARGET_BANDS))])
    selected_bands_resized = np.transpose(selected_bands_resized, (1, 2, 0))  # Rearrange to (512, 512, bands)
    selected_bands_resized = selected_bands_resized.astype(np.float32)
    selected_bands_resized /= np.max(selected_bands_resized, axis=(0, 1), keepdims=True)  # Normalize per band
    return selected_bands_resized

# Load images
rgb_images = np.array([load_rgb_image(f) for f in rgb_files])
hyper_images = [load_hyperspectral_image(f) for f in hyper_files]
hyper_images = np.array([h for h in hyper_images if h is not None])  # Remove any None values

# Ensure equal number of samples before splitting
min_samples = min(len(rgb_images), len(hyper_images))
rgb_images = rgb_images[:min_samples]
hyper_images = hyper_images[:min_samples]

# Save the arrays for training
np.save('rgb_images.npy', rgb_images)
np.save('hyper_images.npy', hyper_images)