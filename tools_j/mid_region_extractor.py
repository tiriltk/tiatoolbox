import glob
from natsort import natsorted
from PIL import Image
from pathlib import Path

def create_wsi(tile_path: str | Path, patch_save_path: Path):
    # Find all images with .png
    image_list = glob.glob(tile_path + '*.png')
    image_list = natsorted(image_list)
    # List with cropped regions of original image
    region_list = []

    # Loop through images
    for i in range(len(image_list)):
        image = Image.open(image_list[i])

        # Define the region to extract (x, y, width, height)
        x = 50           # X coordinate of the left bound
        y = 50           # Y coordinate of the upper bound
        width = 1948     # Width of the region
        height = 1948    # Height of the region
        
        number_x = 11     # Number of patches along x axis
        number_y = 2      # Number of patches along y axis
        # Crop the image to the specified region
        region = image.crop(box= [x, y, x+width, y+height])
        # print(region.height)
        region_list.append(region)
        
        ## Save cropped patches 
        # region.write_to_file("/media/.../" + f"extracted_region_{i}.png")

    # Create a new image with a size to fit all tiles
    new_image = Image.new('RGB', (region_list[0].width * number_x, region_list[0].height * number_y))

    # Loop through the images and paste them into the new image
    for idx, img in enumerate(region_list):
        # Calculate the x and y position for the current image
        x_position = (idx % number_x) * region_list[0].width
        y_position = (idx // number_x) * region_list[0].height
        print(f"x={x_position} for {idx}")
        print(f"y={y_position} for {idx}")
        # Paste the image onto the new image
        new_image.paste(img, (x_position, y_position))

    remove_pixels_right = 106  # Pixels to remove from the right
    remove_pixels_bottom = 106  # Pixels to remove from the bottom

    # Calculate the new size after removing specified pixels from right and bottom
    new_image_width = new_image.width - remove_pixels_right
    new_image_height = new_image.height - remove_pixels_bottom

    # Crop the image
    new_image_cropped = new_image.crop(box = [0, 0, new_image_width, new_image_height])

    # Save final result
    if not patch_save_path.exists():
        patch_save_path.mkdir(parents=True) 
        print(f"Directory {patch_save_path} was created")
    new_image_cropped.save( patch_save_path / "whole_image.png")

if __name__ == "__main__":
    # Path for output
    patch_save_path = Path("/media/jenny/PRIVATE_USB/MM_HE_results_patches/HE_MM009_B_270125_20x_BF_01/Saga/2048x2048/overlay/test_put_together/")
    # Path to tiles
    tile_path = "/media/jenny/PRIVATE_USB/MM_HE_results_patches/HE_MM009_B_270125_20x_BF_01/Saga/2048x2048/overlay/test_put_together/"
    # Call function
    create_wsi(tile_path, patch_save_path)