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
output_dir = "/media/jenny/PRIVATE_USB/co_reg/HE_MM009_2_270125_20x_BF_01/CD8/"

def rmdir(dir_path: str | Path) -> None:
    """Helper function to delete directory."""
    if Path(dir_path).is_dir():
        shutil.rmtree(dir_path)
        logger.info("Removing directory %s", dir_path)


rmdir(global_save_dir)  # remove  directory if it exists from previous runs
global_save_dir.mkdir()
logger.info("Creating new directory %s", global_save_dir)

# fixed_img_file_name = global_save_dir / "fixed_image.tif"
# moving_img_file_name = global_save_dir / "moving_image.tif"

# # Downloading fixed image from COMET dataset
# r = requests.get(
#     "https://tiatoolbox.dcs.warwick.ac.uk/testdata/registration/CRC/06-18270_5_A1MLH1_1.tif",
#     timeout=120,  # 120s
# )
# with fixed_img_file_name.open("wb") as f:
#     f.write(r.content)

# # Downloading moving image from COMET dataset
# r = requests.get(
#     "https://tiatoolbox.dcs.warwick.ac.uk/testdata/registration/CRC/06-18270_5_A1MSH2_1.tif",
#     timeout=120,  # 120s
# )
# with moving_img_file_name.open("wb") as f:
#     f.write(r.content)

# Image paths
fixed_img_file_name = "/media/jenny/PRIVATE_USB/warwick_colab/Pyramidal_TIFF_files/HE_MM009_2_270125_20x_BF_01/Pyramidal_HE_MM009_2_270125_20x_BF_01.tif"
moving_img_file_name = "/media/jenny/PRIVATE_USB/warwick_colab/Pyramidal_TIFF_files/HE_MM009_2_270125_20x_BF_01/Pyramidal_Image_MM009_2_CD8.tif"

fixed_wsi_reader = WSIReader.open(input_img=fixed_img_file_name)
# fixed_image_rgb = fixed_wsi_reader.slide_thumbnail(resolution=0.1563, units="power")
fixed_image_rgb = fixed_wsi_reader.slide_thumbnail(resolution=0.4, units="power")
moving_wsi_reader = WSIReader.open(input_img=moving_img_file_name)
# moving_image_rgb = moving_wsi_reader.slide_thumbnail(resolution=0.1563, units="power")
moving_image_rgb = moving_wsi_reader.slide_thumbnail(resolution=0.4, units="power")

# moving_image_rgb = cv2.resize(moving_image_rgb, (fixed_image_rgb.shape[1], fixed_image_rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
fixed_image_rgb = cv2.resize(fixed_image_rgb, (moving_image_rgb.shape[1], moving_image_rgb.shape[0]), interpolation=cv2.INTER_CUBIC) # Makes sure fixed and moving image have same dimensions

print(fixed_image_rgb)
_, axs = plt.subplots(1, 2, figsize=(15, 8))
axs[0].imshow(fixed_image_rgb, cmap="gray")
axs[0].set_title("Fixed Image")
axs[1].imshow(moving_image_rgb, cmap="gray")
axs[1].set_title("Moving Image")
plt.tight_layout()
plt.savefig(output_dir + "OG_fixed_moving_images.png")
plt.show()


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Pre-process image for registration using masks.

    This function converts the RGB image to grayscale image and
    improves the contrast by linearly rescaling the values.

    """
    image = color.rgb2gray(image)
    image = exposure.rescale_intensity(
        image,
        in_range=tuple(np.percentile(image, (0.5, 99.5))),
    )
    image = image * 255
    return image.astype(np.uint8)


# Preprocessing fixed and moving images
fixed_image = preprocess_image(fixed_image_rgb)
moving_image = preprocess_image(moving_image_rgb)
fixed_image, moving_image = match_histograms(fixed_image, moving_image)

# Visualising the results
_, axs = plt.subplots(1, 2, figsize=(15, 8))
axs[0].imshow(fixed_image, cmap="gray")
axs[0].set_title("Fixed Image")
axs[1].imshow(moving_image, cmap="gray")
axs[1].set_title("Moving Image")
plt.show()

temp = np.repeat(np.expand_dims(fixed_image, axis=2), 3, axis=2)
_saved = cv2.imwrite(str(global_save_dir / "fixed.png"), temp)
temp = np.repeat(np.expand_dims(moving_image, axis=2), 3, axis=2)
_saved = cv2.imwrite(str(global_save_dir / "moving.png"), temp)


save_dir = global_save_dir / "tissue_mask"
if save_dir.exists():
    shutil.rmtree(save_dir, ignore_errors=False, onexc=None)

segmentor = SemanticSegmentor(
    pretrained_model="unet_tissue_mask_tsef",
    num_loader_workers=4,
    batch_size=1,
)

output = segmentor.predict(
    [
        global_save_dir / "fixed.png",
        global_save_dir / "moving.png",
    ],
    save_dir=save_dir,
    mode="tile",
    resolution=1.0,
    units="baseline",
    patch_input_shape=[1024, 1024],
    patch_output_shape=[512, 512],
    stride_shape=[512, 512],
    device=device,
    crash_on_exception=True,
)

def post_processing_mask(mask: np.ndarray) -> np.ndarray:
    """Post-process WSI masks."""
    mask = ndimage.binary_fill_holes(mask, structure=np.ones((3, 3))).astype(int)

    # num of unique objects for segmentation is 2.
    num_unique_labels = 2
    # remove all the objects while keep the biggest object only
    label_img = measure.label(mask)
    if len(np.unique(label_img)) > num_unique_labels:
        regions = measure.regionprops(label_img)
        mask = mask.astype(bool)
        all_area = [i.area for i in regions]
        second_max = max([i for i in all_area if i != max(all_area)])
        mask = morphology.remove_small_objects(mask, min_size=second_max + 1)
    return mask.astype(np.uint8)


fixed_mask = np.load(output[0][1] + ".raw.0.npy")
moving_mask = np.load(output[1][1] + ".raw.0.npy")
# moving_mask = Image.open("/media/jenny/PRIVATE_USB/aSMA/SMA/SMA_Analysis/Image_MM266_2A_SMA_BW_0_4.tif")
# moving_mask = moving_mask.convert("L")
# moving_mask = np.array(moving_mask)
# print(moving_mask.dtype)
# print(moving_mask)
# moving_mask = cv2.resize(moving_mask, (moving_image_rgb.shape[1], moving_image_rgb.shape[0]), interpolation=cv2.INTER_CUBIC)


# num of unique objects for segmentation is 2.
num_unique_labels = 2

# Simple processing of the raw prediction to generate semantic segmentation task
fixed_mask = np.argmax(fixed_mask, axis=-1) == num_unique_labels
moving_mask = np.argmax(moving_mask, axis=-1) == num_unique_labels

fixed_mask = post_processing_mask(fixed_mask)
moving_mask = post_processing_mask(moving_mask)

_, axs = plt.subplots(1, 2, figsize=(15, 8))
axs[0].imshow(fixed_mask, cmap="gray")
axs[0].set_title("Fixed Mask")
axs[1].imshow(moving_mask, cmap="gray")
axs[1].set_title("Moving Mask")
plt.tight_layout()
plt.savefig(output_dir + "fixed_moving_masks.png")
plt.show()

dfbr_fixed_image = np.repeat(np.expand_dims(fixed_image, axis=2), 3, axis=2)
dfbr_moving_image = np.repeat(np.expand_dims(moving_image, axis=2), 3, axis=2)

dfbr = DFBRegister()
dfbr_transform = dfbr.register(
    dfbr_fixed_image,
    dfbr_moving_image,
    fixed_mask,
    moving_mask,
)
np.save(output_dir + "affine_trans_matrix", dfbr_transform)
# # Visualization
# original_moving = cv2.warpAffine(
#     moving_image,
#     np.eye(2, 3),
#     fixed_image.shape[:2][::-1],
# )
# dfbr_registered_image = cv2.warpAffine(
#     moving_image,
#     dfbr_transform[0:-1],
#     fixed_image.shape[:2][::-1],
# )
# dfbr_registered_mask = cv2.warpAffine(
#     moving_mask,
#     dfbr_transform[0:-1],
#     fixed_image.shape[:2][::-1],
# )

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
dfbr_registered_mask = cv2.warpAffine(
    moving_mask,
    dfbr_transform[0:-1],
    fixed_image.shape[:2][::-1],
)
print(f"original_moving.shape{original_moving.shape}")
print(f"fixed_image.shape{fixed_image.shape}")
print(f"dfbr_registered_image.shape{dfbr_registered_image.shape}")
print(f"moving_image_rgb.shape{moving_image_rgb.shape}")
# before_overlay = np.dstack((original_moving, fixed_image, original_moving))
# # dfbr_overlay = np.dstack((dfbr_registered_image, fixed_image, dfbr_registered_image))

# _, axs = plt.subplots(1, 2, figsize=(15, 8))
# axs[0].imshow(before_overlay)
# axs[0].set_title("Overlay Before Registration")
# axs[1].imshow(fixed_image)
# axs[1].set_title("Overlay After DFBR")
# plt.tight_layout()
# plt.savefig(output_dir + "DFBR_overlay.png")
# plt.show()

# Create an overlay image with opaque representation
alpha = 0.5

# Overlay where the two images are blended based on alpha
before_overlay = cv2.addWeighted(fixed_image_rgb, alpha, original_moving, alpha, 0)

dfbr_overlay = cv2.addWeighted(fixed_image_rgb, alpha, dfbr_registered_image, alpha, 0)

# cv2.imwrite(output_dir + "DFBR_overlay.png", cv2.cvtColor(dfbr_overlay, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
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