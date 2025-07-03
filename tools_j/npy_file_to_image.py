import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

def save_image_patches(file_path: str | Path, save_path: str | Path):
    
    # Load images
    images = np.load(file_path)
    print("Array Shape:", images.shape)
    print("Data Type:", images.dtype)
    print("Max value:", np.max(images))
    print("Min value:", np.min(images))

    # Ensure the pixel values are within the 0-255 range and are integers
    assert images.max() <= 255 and images.min() >= 0, "Pixel values are out of the expected range."

    # Create the output directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    num_images_to_save = min(2523, images.shape[0])

    for i in range(num_images_to_save):
        # Convert the image data type to uint8 if it is not already
        image_uint8 = images[i].astype(np.uint8)
        
        # Save image
        plt.imsave(os.path.join(save_path, f'image_{i+1}.png'), image_uint8)  
        print(f"Image {i+1} saved as image_{i+1}.png")

if __name__ == "__main__":
    file_path = '/media/jenny/PRIVATE_USB/AugHoverData/data/images.npy'
    save_path = '/media/jenny/PRIVATE_USB/AugHoverData/all_conic_images/'
    save_image_patches(file_path, save_path)