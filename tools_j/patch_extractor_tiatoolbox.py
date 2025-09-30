import logging

if logging.getLogger().hasHandlers():
    logging.getLogger().handlers.clear()

from pathlib import Path
import os
from PIL import Image
import sys

import tifffile

# Get the path to the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Append the project root to sys.path
if project_root not in sys.path:
    sys.path.append(project_root)

from tiatoolbox.wsicore import WSIReader
from tiatoolbox.tools.patchextraction import SlidingWindowPatchExtractor


def extract_patches_wsi(wsi_path, patch_save_path, mask_path = None):

    wsi = WSIReader.open(wsi_path)
    dim = wsi.slide_dimensions(resolution=1, units="power")
    # Print pixels along x- and y axis for wsi
    print(f"wsi dim: {dim}")

    nr_extra_pixels = 0          # Number of extra pixels per patch
    pixels = 2048                # Pixels along x- and y- axis for each patch before extra pixels are added. x and y must have same number of pixels in numbers of 2^x if processed with aug_hovernet.
    nr_pixels_overlap = 100     # Number of pixels overlap between patches
    # number of pixels along x- and y axis for each patch
    x = pixels + nr_extra_pixels
    y = pixels + nr_extra_pixels

    extractor = SlidingWindowPatchExtractor(
        wsi_path,
        patch_size=(x, y),
        stride=(x-nr_pixels_overlap, y-nr_pixels_overlap),
        # mask_path

    )
    print(f"Number of images: {len(extractor)}")

    # Iterate over the patches and save each patch
    for patch_idx in range(len(extractor)): 
        patch = extractor[patch_idx]

        patch_image = Image.fromarray(patch)
        patch_filename = os.path.join(patch_save_path, f"patch_{patch_idx}.png")
        patch_image.save(patch_filename)

def extract_patches(image_path, patch_save_path):
    img = tifffile.imread(image_path)

    nr_extra_pixels = 0        # Number of extra pixels per patch
    pixels = 2048               # Pixels along x- and y- axis for each patch before extra pixels are added. x and y must have same number of pixels if processed with aug_hovernet.
    nr_pixels_overlap = 100     # Number of pixels overlap between patches
    # number of pixels along x- and y axis for each patch
    x = pixels + nr_extra_pixels
    y = pixels + nr_extra_pixels

    extractor = SlidingWindowPatchExtractor(
        img,
        patch_size=(x, y),
        stride=(x-nr_pixels_overlap, y-nr_pixels_overlap)
    )
    print(f"Number of images: {len(extractor)}")

    # Iterate over the patches and save each patch
    for patch_idx in range(len(extractor)): 
        patch = extractor[patch_idx]

        patch_image = Image.fromarray(patch)
        patch_filename = os.path.join(patch_save_path, f"patch_{patch_idx}.png")
        patch_image.save(patch_filename)

if __name__ == "__main__":

    # wsi_path = "/media/.../Pyramidal_TIFF_files/.../Pyramidal_wsi.tif"
    image_path = "/Volumes/Expansion/HE/20x/masks/Func116_ST_HE_20x_BF_01_black_mask.tif"
    #wsi_path = "/Volumes/Expansion/HE/20x/images/Func116_ST_HE_20x_BF_01.tif"
    # patch_save_path = Path("/media/jenny/Expansion/MM_HE_patches/HE_MM179_B_70225_20x_BF_01/aughovernet/2048x2048/")
    # image_path = "/media/jenny/Expansion/MM_HE_masks/HE_MM179_2C_290125_20x_BF_01/HE_MM179_2C_290125_20x_BF_01_no_folds_mask.png"
    patch_save_path = Path("/Volumes/Expansion/HE_patches/20x/Func116_ST_HE_20x_BF_01/aughovernet/2048x2048_mask/")
    if not patch_save_path.exists():
        patch_save_path.mkdir(parents=True)
        print(f"Directory {patch_save_path} was created")
    
    # mask_path = "/media/.../masks/.../image.jpg"

    #extract_patches_wsi(wsi_path, patch_save_path)
    extract_patches(image_path, patch_save_path)
