import pyvips
from PIL import Image
from pathlib import Path

def convert_tiff(image_path : str | Path, save_path : str | Path):
    # Load the original TIFF file
    image = pyvips.Image.tiffload(image_path)
    # Save as a pyramidal tiled TIFF
    image.tiffsave(save_path,
                tile=True,
                pyramid=True,
                compression='deflate',  # or 'lzw', etc.
                tile_width=256,
                tile_height=256)  

def convert_and_resize_tiff(image_path : str | Path, save_path : str | Path):
    # Load the original TIFF file
    image = pyvips.Image.tiffload(image_path)
    # Resize image to 40x
    image = image.resize(2, kernel = "lanczos3") # Factor to scale image by and resampling kernel 
    # Save as a pyramidal tiled TIFF
    image.tiffsave(save_path,
                tile=True,
                pyramid=True,
                compression='deflate',  # or 'lzw', etc.
                tile_width=256,
                tile_height=256)  


if __name__ == "__main__":
    # Path to TIFF file 
    image_path = "media/.../image.tif"
    save_path = "/media/.../Pyramidal_TIFF_files/.../Pyramidal_image.tif"
    convert_tiff(image_path, save_path)
    # convert_and_resize_tiff(image_path, save_path)
