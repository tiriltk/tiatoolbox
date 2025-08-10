import os
import numpy as np
from pathlib import Path
from skimage import io, color, filters, morphology
import pyvips
def create_mask(wsi_path : str | Path, wsi_name : str, mask_path_save : str | Path):
    # # Read image and apply grayscale
    # color_img = io.imread(wsi_path)
    # gray_img = color.rgb2gray(color_img)
    # Load RGB image
    img = pyvips.Image.new_from_file(wsi_path, access="sequential")

    # Convert to grayscale
    gray = img.colourspace("b-w")  # single-band grayscale

    # Convert to NumPy
    gray_np = np.ndarray(buffer=gray.write_to_memory(),
                     dtype=np.uint8,
                     shape=(gray.height, gray.width))
    gray_img = gray_np.astype(np.float32) / 255.0

    # Threshold segmentation on grayscale image
    print("Start treshhold --------------------------------")
    threshold = filters.threshold_otsu(gray_img)
    binary_img = gray_img > threshold
    print("Finished treshhold -----------------------------")

    # Morphological operations to remove small artifacts and smooth the binary image
    selem = morphology.disk(20)
    print("Start opening and closing-----------------------")
    binary_img = morphology.opening(binary_img, selem)
    binary_img = morphology.closing(binary_img, selem)
    print("Finished opening and closing -------------------")
    binary_img = np.logical_not(binary_img)
    binary_img = morphology.remove_small_objects(binary_img, min_size=50000)
    print("Objects removed --------------------------------")
    
    # Write to file
    roi_file_path = os.path.join(mask_path_save, wsi_name + '.png')
    io.imsave(roi_file_path, binary_img.astype(np.uint8) * 255)

if __name__ == "__main__":
    wsi_path = "/media/jenny/Expansion/tissue_fold/HE_MM009_B_270125_20x_BF_01_no_fold.tif"
    wsi_name = "HE_MM009_B_270125_no_fold_20"
    mask_path_save = "/media/jenny/Expansion/MM_HE_masks/test/"
    
    create_mask(wsi_path, wsi_name, mask_path_save)