# import torch
# import os
# checkpoint = torch.load("../../../../../media/jenny/PRIVATE_USB/Hover_net_files/Pannuke_checkpoints/hovernet_fast_pannuke_type_tf2pytorch.tar")
# # print(checkpoint.keys())  # Lists the stored keys
# desc = checkpoint['desc']
# # print(checkpoint["desc"])
# # print(desc.keys())
# for key, value in desc.items():
#     print(f"Key: {key}")
#     print(f"Value (tensor): {value}")
#     print(f"Tensor shape: {value.shape}")  # This will show the shape of the tensor
#     print("-" * 50)  #

# print(torch.cuda.is_available())  # Should return True if CUDA is accessible
# print(torch.version.cuda)  # Shows the CUDA version PyTorch was built with

# file_path = "/home/jenny/Hover/hover_net/IMG_1370.jpg"
# print(os.path.exists(file_path))

import openslide
import xml.etree.ElementTree as ET
import tifffile


# slide = openslide.OpenSlide('/media/jenny/PRIVATE_USB/Converted_images/Pyramidal_HE_MM009_2_270125_20x_BF_01.tif')
# properties = slide.properties
# # for prop in properties:
# #     print(prop, properties[prop])
# # Access and print the OBJECTIVE_POWER property if it exists
# objective_power = properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)

def extract_nominal_magnification(file_path):
    # Open the TIFF file to access ImageDescription
    with tifffile.TiffFile(file_path) as tif:
        # Assuming the OME-XML metadata is in the first page
        image_description = tif.pages[0].description

    # Parse the XML data
    try:
        root = ET.fromstring(image_description)
        namespaces = {'OME': 'http://www.openmicroscopy.org/Schemas/OME/2015-01'}
        
        # Find the <OME:Objective> element
        objective = root.find('.//OME:Objective', namespaces)
        
        # Extract the NominalMagnification attribute
        if objective is not None:
            nominal_magnification = objective.get('NominalMagnification')
            if nominal_magnification is not None:
                return float(nominal_magnification)
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")

    # If extraction fails, return None or raise an exception
    print("NominalMagnification not found.")
    return None

# file_path = '/media/jenny/PRIVATE_USB/Converted_images/Pyramidal_HE_MM009_2_270125_20x_BF_01.tif'
# nominal_magnification = extract_nominal_magnification(file_path)
# if nominal_magnification is not None:
#     print(f"Nominal Magnification: {nominal_magnification}")

def extract_physical_size(file_path):
    # Open the TIFF file to access ImageDescription
    with tifffile.TiffFile(file_path) as tif:
        # Assuming the OME-XML metadata is in the first page
        image_description = tif.pages[0].description

    # Parse the XML data
    try:
        root = ET.fromstring(image_description)
        namespaces = {'OME': 'http://www.openmicroscopy.org/Schemas/OME/2015-01'}
        
        # Access the <OME:Pixels> element
        pixels = root.find('.//OME:Pixels', namespaces)
        
        # Extract PhysicalSizeX and PhysicalSizeY
        if pixels is not None:
            physical_size_x = pixels.get('PhysicalSizeX')
            physical_size_y = pixels.get('PhysicalSizeY')
            if physical_size_x is not None and physical_size_y is not None:
                return float(physical_size_x), float(physical_size_y)
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")

    # If extraction fails, return None or raise an appropriate message
    print("PhysicalSizeX or PhysicalSizeY not found.")
    return None, None

# file_path = '/media/jenny/PRIVATE_USB/Converted_images/Pyramidal_HE_MM009_2_270125_20x_BF_01.tif'
# mpp_x, mpp_y = extract_physical_size(file_path)
# if mpp_x is not None and mpp_y is not None:
#     print(f"Microns per pixel - X: {mpp_x}, Y: {mpp_y}")