import glob 
from natsort import natsorted
from PIL import Image
from pathlib import Path

def merge_images(tile_path : str | Path, save_path : str | Path):
    image_list = glob.glob(tile_path + '*.png')
    image_list = natsorted(image_list)
    imgs_load_list = []

    # Load images
    for i in range(len(image_list)):
        image_fetch = image_list[i]
        image = Image.open(image_fetch)
        imgs_load_list.append(image)
    
    number_x = 4      # Number of patches along x axis
    number_y = 2      # Number of patches along y axis

    # Create a new image with a size to fit all tiles
    new_image = Image.new('RGB', (imgs_load_list[0].width * number_x, imgs_load_list[0].height * number_y))

    # Loop through the images and paste them into the new image
    for idx, img in enumerate(imgs_load_list):
        # Calculate the x and y position for the current image
        x_position = (idx % number_x) * imgs_load_list[0].width
        y_position = (idx // number_y) * imgs_load_list[0].height
        
        # Paste the image onto the new image
        new_image.paste(img, (x_position, y_position))

    new_image.save(save_path)

if __name__ == "_main":
    # Path to tiles
    tile_path = "/media/.../"
    save_path = "/media/.../512x512_part_1.png"
    merge_images(tile_path, save_path)
