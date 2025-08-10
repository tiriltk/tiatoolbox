import pyvips
from pathlib import Path

def create_thumbnail(input_tiff_path : str | Path, output_thumbnail_path : str | Path):
    # Load the TIFF image
    image = pyvips.Image.new_from_file(input_tiff_path, access='sequential')

    # Calculate the scale needed to resize the image
    scale = 0.2
    # Resize the image to the thumbnail size
    thumbnail = image.resize(scale)

    # Save thumbnail
    thumbnail.write_to_file(output_thumbnail_path)

if __name__ == "__main__":
    # Input and output path
    input_tiff_path = "/media/jenny/Expansion/MM_HE_results/HE_MM179_2D_290125_20x_BF_01/mask/wsi/whole_image_complete.png"
    # input_tiff_path = "/media/jenny/Expansion/aSMA/Image_MM179_A_SMA.tif"
    # input_tiff_path = "/media/jenny/Expansion/MM_HE_20x_TIFF/HE_MM179_A_70225_vsi_Collection/HE_MM179_A_70225_20x_BF_01/HE_MM179_A_70225_20x_BF_01.tif"
    output_thumbnail_path = "/media/jenny/Expansion/MM_HE_thumbnail/MM179_2D/result_HE_MM179_2D_290125_20x_BF_01.png"
    

    create_thumbnail(input_tiff_path, output_thumbnail_path)