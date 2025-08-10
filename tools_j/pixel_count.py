import pyvips
import numpy as np

def count_black_pixels(image_path):
    # Load image with pyvips
    img = pyvips.Image.new_from_file(image_path, access="sequential") 

    # Ensure image is 8-bit
    if img.format != "uchar":
        img = img.cast("uchar")

    # Convert to NumPy for pixel-wise operation
    np_img = np.ndarray(
        buffer=img.write_to_memory(),
        dtype=np.uint8,
        shape=(img.height, img.width, img.bands)
    )

    # Mask for black pixels
    black_mask = np.all(np_img[:, :, :3] != 255, axis=2)
    black_pixel_count = np.sum(black_mask)

    return black_pixel_count

if __name__ == "__main__":
    image_path = "/media/jenny/Expansion/MM_HE_masks/test/HE_MM009_B_270125_20x_BF_01_no_fold.png"
    count = count_black_pixels(image_path)
    print(f"Black pixel count: {count}")
