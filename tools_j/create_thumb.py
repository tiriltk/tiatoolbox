import pyvips
from pathlib import Path

def create_thumbnail(input_tiff_path : str | Path, output_thumbnail_path : str | Path, thumbnail_width : int):
    # Load the TIFF image
    image = pyvips.Image.new_from_file(input_tiff_path, access='sequential')

    # Calculate the scale needed to resize the image
    scale = thumbnail_width / image.width
    
    # Resize the image to the thumbnail size
    thumbnail = image.resize(scale)

    # Save thumbnail
    thumbnail.write_to_file(output_thumbnail_path)

if __name__ == "__main__":
    # Input and output path
    input_tiff_path = "/media/.../Pyramidal_TIFF_files/.../Pyramidal_image.tif"
    output_thumbnail_path = "/media/.../thumbnail/Pyramidal_image.png"
    thumbnail_width = 2048  # Width of thumbnail

    create_thumbnail(input_tiff_path, output_thumbnail_path, thumbnail_width)