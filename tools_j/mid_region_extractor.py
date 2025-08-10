import glob
from natsort import natsorted
# from PIL import Image
# Image.MAX_IMAGE_PIXELS = None  # Set to None to disable the limit
from pathlib import Path
import pyvips
import matplotlib.pyplot as plt

def create_wsi(tile_path: str | Path, patch_save_path: Path, wsi_path: str | Path):
    wsi_image = pyvips.Image.new_from_file(wsi_path)
    missing_pixels = 50
    # Regions in WSI that were cropped out when creating patches. Needs to be pasted into final image.
    region_wsi_lenght = wsi_image.crop(0, 0, wsi_image.width, missing_pixels)
    region_wsi_height = wsi_image.crop(0, 0, missing_pixels, wsi_image.height)

    # Find all images with .png
    image_list = glob.glob(tile_path + '*.png')
    image_list = natsorted(image_list)
    # List with cropped regions of original image
    region_list = []

    # Loop through images
    for i in range(len(image_list)):
        image = pyvips.Image.new_from_file(image_list[i])
        # Define the region to extract (x, y, width, height)
        x = 50           # X coordinate of the left bound
        y = 50           # Y coordinate of the upper bound
        width = 1948     # Width of the region
        height = 1948    # Height of the region
        
        number_x = 11     # Number of patches along x axis
        number_y = 11     # Number of patches along y axis
        region = image.crop(x, y, width, height)
        region_list.append(region)
    
        ## Save cropped patches 
        # region.write_to_file("/media/jenny/Expansion/MM_HE_results/HE_MM179_2D_290125_20x_BF_01/wsi/tiles/" + f"extracted_region_{i}.png")

    # Create a new image with a size to fit all tiles
    new_image = pyvips.Image.black(region_list[0].width * number_x + missing_pixels, region_list[0].height * number_y + missing_pixels, bands = 3)
    # Insert leftmost and uppermost cropped out pixels from original wsi file
    new_image = new_image.insert(region_wsi_lenght, 0, 0)
    new_image = new_image.insert(region_wsi_height, 0, 0)

    # Loop through the images and paste them into the new image
    for idx, img in enumerate(region_list):
        # Calculate the x and y position for the current image
        x_position = (idx % number_x) * region_list[0].width + missing_pixels
        y_position = (idx // number_x) * region_list[0].height + missing_pixels
        new_image=new_image.insert(img, x_position, y_position)
    
    remove_pixels_right = new_image.width - wsi_image.width # Pixels to remove from the right
    # remove_pixels_right = 0
    remove_pixels_bottom = new_image.height - wsi_image.height # Pixels to remove from the bottom
    # print(new_image.width)
    # print(new_image.height)
    # print(wsi_image.width)
    # print(wsi_image.height)
    # print(remove_pixels_bottom)
    # print(remove_pixels_right)

    # Calculate the new size after removing specified pixels from right and bottom
    new_image_width = new_image.width - remove_pixels_right
    new_image_height = new_image.height - remove_pixels_bottom
    print(f"Output image width: {new_image_width}")
    print(f"Output image height: {new_image_height}")
    # Crop the image
    new_image_cropped = new_image.crop(0, 0, new_image_width, new_image_height)

    # Save final result
    if not patch_save_path.exists():
        patch_save_path.mkdir(parents=True) 
        print(f"Directory {patch_save_path} was created")
    new_image_cropped.write_to_file( patch_save_path / "whole_image_complete.png")

if __name__ == "__main__":
    # Path for output
    patch_save_path = Path("/media/jenny/Expansion/MM_HE_results/HE_MM009_B_270125/2048x2048/mask/wsi/")
    # Path to tiles
    tile_path = "/media/jenny/Expansion/MM_HE_results/HE_MM009_B_270125/2048x2048/mask/overlay/"
    # Path to wsi
    wsi_path = "/media/jenny/Expansion/MM_HE_pyramidal_tiff/Pyramidal_HE_MM009_B_270125.tif"
    # Call function
    create_wsi(tile_path, patch_save_path, wsi_path)

# First attempt, but improved version above
# def create_wsi(tile_path: str | Path, patch_save_path: Path):
#     # Find all images with .png
#     image_list = glob.glob(tile_path + '*.png')
#     image_list = natsorted(image_list)
#     # List with cropped regions of original image
#     region_list = []

#     # Loop through images
#     for i in range(len(image_list)):
#         image = Image.open(image_list[i])

#         # Define the region to extract (x, y, width, height)
#         x = 50           # X coordinate of the left bound
#         y = 50           # Y coordinate of the upper bound
#         width = 1948     # Width of the region
#         height = 1948    # Height of the region
        
#         number_x = 11     # Number of patches along x axis
#         number_y = 11     # Number of patches along y axis
#         # Crop the image to the specified region
#         region = image.crop(box= [x, y, x+width, y+height])
#         # print(region.height)
#         region_list.append(region)
        
#         ## Save cropped patches 
#         # region.write_to_file("/media/.../" + f"extracted_region_{i}.png")

#     # Create a new image with a size to fit all tiles
#     new_image = Image.new('RGB', (region_list[0].width * number_x, region_list[0].height * number_y))


#     # Loop through the images and paste them into the new image
#     for idx, img in enumerate(region_list):
#         # Calculate the x and y position for the current image
#         x_position = (idx % number_x) * region_list[0].width
#         y_position = (idx // number_x) * region_list[0].height
#         print(f"x={x_position} for {idx}")
#         print(f"y={y_position} for {idx}")
#         # Paste the image onto the new image
#         new_image.paste(img, (x_position, y_position))

#     remove_pixels_right = 106  # Pixels to remove from the right
#     remove_pixels_bottom = 800  # Pixels to remove from the bottom

#     # Calculate the new size after removing specified pixels from right and bottom
#     new_image_width = new_image.width - remove_pixels_right
#     new_image_height = new_image.height - remove_pixels_bottom

#     # Crop the image
#     new_image_cropped = new_image.crop(box = [0, 0, new_image_width, new_image_height])

#     # Save final result
#     if not patch_save_path.exists():
#         patch_save_path.mkdir(parents=True) 
#         print(f"Directory {patch_save_path} was created")
#     new_image.save( patch_save_path / "whole_image.png")

# if __name__ == "__main__":
#     # Path for output
#     patch_save_path = Path("/media/jenny/Expansion/MM_HE_results/HE_MM009_B_270125/2048x2048/wsi/")
#     # Path to tiles
#     tile_path = "/media/jenny/Expansion/MM_HE_results/HE_MM009_B_270125/2048x2048/overlay/"
#     # Call function
#     create_wsi(tile_path, patch_save_path)