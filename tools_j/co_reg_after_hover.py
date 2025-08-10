from __future__ import annotations

# Clear logger to use tiatoolbox.logger
import logging

if logging.getLogger().hasHandlers():
    logging.getLogger().handlers.clear()

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import shutil
import warnings
from pathlib import Path
import sys
# Get the path to the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Append the project root to sys.path
if project_root not in sys.path:
    sys.path.append(project_root)
import cv2
import matplotlib as mpl
import numpy as np
import requests
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import color, exposure, measure, morphology

from tiatoolbox import logger
from tiatoolbox.models.engine.semantic_segmentor import SemanticSegmentor
from tiatoolbox.tools.registration.wsi_registration import (
    AffineWSITransformer,
    DFBRegister,
    apply_bspline_transform,
    estimate_bspline_transform,
    match_histograms,
)
from tiatoolbox.wsicore.wsireader import WSIReader
from PIL import Image
device = "cuda"

warnings.filterwarnings("ignore")
global_save_dir = Path("./tmp/")

def rmdir(dir_path: str | Path) -> None:
    """Helper function to delete directory."""
    if Path(dir_path).is_dir():
        shutil.rmtree(dir_path)
        logger.info("Removing directory %s", dir_path)


rmdir(global_save_dir)  # remove  directory if it exists from previous runs
global_save_dir.mkdir()
logger.info("Creating new directory %s", global_save_dir)

def co_reg(fixed_img_file_name, moving_img_file_name, fixed_image_path, dfbr_transform_path, output_dir): 

    # Make sure image is RGB 
    fixed_image_load = cv2.imread(fixed_img_file_name)
    fixed_image_load = cv2.cvtColor(fixed_image_load, cv2.COLOR_BGR2RGB)
    moving_image_load = cv2.imread(moving_img_file_name)
    moving_image_load = cv2.cvtColor(moving_image_load, cv2.COLOR_BGR2RGB)
    # Extract image shape
    height_f, width_f = fixed_image_load.shape[:2]
    height_m, width_m = moving_image_load.shape[:2]
    thumb_size_f = (int(width_f*0.4), int(height_f*0.4))
    thumb_size_m = (int(width_m*0.4), int(height_m*0.4)+1)

    # Reduce resolution
    fixed_image_rgb = cv2.resize(fixed_image_load, thumb_size_f)
    moving_image_rgb = cv2.resize(moving_image_load, thumb_size_m)

    # moving_image_rgb = cv2.resize(moving_image_rgb, (fixed_image_rgb.shape[1], fixed_image_rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
    fixed_image_rgb = cv2.resize(fixed_image_rgb, (moving_image_rgb.shape[1], moving_image_rgb.shape[0]), interpolation=cv2.INTER_CUBIC) # Makes sure fixed and moving image have same dimensions

    _, axs = plt.subplots(1, 2, figsize=(15, 8))
    axs[0].imshow(fixed_image_rgb, cmap="gray")
    axs[0].set_title("Fixed Image")
    axs[1].imshow(moving_image_rgb, cmap="gray")
    axs[1].set_title("Moving Image")
    plt.tight_layout()
    plt.savefig(output_dir + "OG_fixed_moving_images.png")
    plt.show()

    fixed_image = cv2.imread(fixed_image_path)
    dfbr_transform = np.load(dfbr_transform_path)

    # Visualization
    original_moving = cv2.warpAffine(
        moving_image_rgb,
        np.eye(2, 3),
        fixed_image.shape[:2][::-1],
    )
    dfbr_registered_image = cv2.warpAffine(
        moving_image_rgb,
        dfbr_transform[0:-1],
        fixed_image.shape[:2][::-1],
    )

    print(f"original_moving.shape{original_moving.shape}")
    print(f"fixed_image.shape{fixed_image.shape}")
    print(f"dfbr_registered_image.shape{dfbr_registered_image.shape}")
    print(f"moving_image_rgb.shape{moving_image_rgb.shape}")
    print(f"fixed_image_rgb.shape{fixed_image_rgb.shape}")

    # Create an overlay image with opaque representation
    alpha = 0.5

    # Overlay where the two images are blended based on alpha
    before_overlay = cv2.addWeighted(fixed_image_rgb, alpha, original_moving, alpha, 0)

    dfbr_overlay = cv2.addWeighted(fixed_image_rgb, alpha, dfbr_registered_image, alpha, 0)

    cv2.imwrite(output_dir + "DFBR_overlay.png", cv2.cvtColor(dfbr_overlay, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    cv2.imwrite(output_dir + "fixed_image_rgb.png", cv2.cvtColor(fixed_image_rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    cv2.imwrite(output_dir + "dfbr_registered_image.png", cv2.cvtColor(dfbr_registered_image, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_PNG_COMPRESSION), 3])

    fig, axs = plt.subplots(1, 2, figsize=(15, 8))

    # Show the overlay
    axs[0].imshow(before_overlay)
    axs[0].set_title("Overlay Before Registration")

    axs[1].imshow(dfbr_overlay)
    axs[1].set_title("Overlay After DFBR")

    plt.tight_layout()
    plt.savefig(output_dir + "DFBR_and_before_overlay.png")
    plt.show()


if __name__ == "__main__":
    # Image paths
    fixed_img_file_name = "/media/jenny/Expansion/MM_HE_results/HE_MM009_B_270125/2048x2048/wsi/whole_image_complete.png"
    moving_img_file_name = "/media/jenny/Expansion/co_reg/HE_MM009_2_270125_20x_BF_01/CD8/Pyramidal_Image_MM009_B_CD8.tif"
    
    # Files from co_reg.py
    fixed_image_path = "/media/jenny/Expansion/co_reg/HE_MM009_2_270125_20x_BF_01/CD8/wsi_org_output/adapt/fixed.png"
    dfbr_transform_path = "/media/jenny/Expansion/co_reg/HE_MM009_2_270125_20x_BF_01/CD8/wsi_org_output/adapt/affine_trans_matrix.npy"
    
    # Output directory
    output_dir = "/media/jenny/Expansion/co_reg/HE_MM009_2_270125_20x_BF_01/CD8/hover_output/"


    co_reg(fixed_img_file_name, moving_img_file_name, fixed_image_path, dfbr_transform_path, output_dir)



# # Specified region of WSI
# location = (1000, 1000)  # at base level 0
# size = (500, 500)  # (width, height)

# # Extract region from the fixed whole slide image
# fixed_tile = fixed_wsi_reader.read_rect(location, size, resolution=2.5, units="power")

# # DFBR transform is computed for level 6
# # Hence it should be mapped to level 0 for AffineWSITransformer
# dfbr_transform_level = 6
# transform_level0 = dfbr_transform * [
#     [1, 1, 2**dfbr_transform_level],
#     [1, 1, 2**dfbr_transform_level],
#     [1, 1, 1],
# ]

# # Extract transformed region from the moving whole slide image
# tfm = AffineWSITransformer(moving_wsi_reader, transform_level0)
# moving_tile = tfm.read_rect(location, size, resolution=2.5, units="power")

# _, axs = plt.subplots(1, 2, figsize=(15, 8))
# axs[0].imshow(fixed_tile, cmap="gray")
# axs[0].set_title("Fixed Tile")
# axs[1].imshow(moving_tile, cmap="gray")
# axs[1].set_title("Moving Tile")
# plt.show()