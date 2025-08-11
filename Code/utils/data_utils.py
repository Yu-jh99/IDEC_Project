from PIL import Image
import numpy as np
import os

def is_image_file(filename):
    lower = filename.lower()
    return any(lower.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"])

def load_and_normalize(img_path, mean, std):
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0  # HWC
    arr = (arr - mean[None, None, :]) / std[None, None, :]
    return arr
#Optionally: inverse normalization for visualization
def denormalize(arr, mean, std):
	arr = arr * std[None, None, :] + mean[None, None, :]
	arr = np.clip(arr, 0.0, 1.0)
	return (arr * 255).astype(np.unit8)
