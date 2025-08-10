import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import glob 
from natsort import natsorted

def read_metrics_data(file_path, encoding):
    counts_list = glob.glob(file_path + '*.csv')
    counts_list = natsorted(counts_list)
    # Get column order from the first file
    first_df = pd.read_csv(counts_list[0])
    column_order = first_df.columns.tolist()

    # Initialize a DataFrame to accumulate sums
    total_sums = None
    
    for file in counts_list:
        # pd.set_option('display.max_rows', None)     # Show all rows
        # pd.set_option('display.max_columns', None)  # Show all columns
        # pd.set_option('display.width', 1000)        # Adjust the width of the display
        # Read metrics from CSV file
        df = pd.read_csv(file, encoding=encoding)
        # Convert everything to numeric where possible
        numeric_df = df.apply(pd.to_numeric, errors='coerce')
        
        # Sum columns for this file
        file_sums = numeric_df.sum()

        if total_sums is None:
            total_sums = file_sums
        else:
            total_sums = total_sums.add(file_sums, fill_value=0)

    # Reorder to match original column order
    total_sums = total_sums.reindex(column_order).fillna(0).astype(int)
    total_sums = total_sums.drop(column_order[0])
    
    # Convert to a single-row DataFrame
    result_df = pd.DataFrame([total_sums])

    # print(result_df)
    # result_df.to_csv('/media/jenny/Expansion/MM_HE_patches/HE_MM009_B_270125/aughovernet/debug_test_tiles/tiles_result_csv_4/pixel_count.csv', index=False)
    return result_df
        

def read_metrics_data_all_combined(file_path, encoding):
    """Read metrics from CSV file."""
    # Adjust to show entire DataFrame
    pd.set_option('display.max_rows', None)     # Show all rows
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', 1000)        # Adjust the width of the display
    
    # Initialize an empty list to hold DataFrames
    data_frames = []

    for i in range(40, 47):
        # Construct the full file name
        print(i)
        file_name = f"{file_path}{i}.csv"
        
        # Read the CSV file
        df = pd.read_csv(file_name, encoding=encoding)
        
        # Append the DataFrame to the list
        data_frames.append(df)

    # Concatenate all DataFrames into one
    combined_df = pd.concat(data_frames, ignore_index=True)

    # Reset display options back to default if needed
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')

    # Print the combined DataFrame
    print(combined_df)
    
    return combined_df

def load_npy_images(npys_path):
    """Load images from a .npy file."""
    images = np.load(npys_path)
    return images  # Expect images to be in shape (num_images, height, width, channels)

if __name__ == "__main__":

    # Read the CSV and return dataframe
    csv_file_path = "/media/jenny/Expansion/MM_HE_results/HE_MM009_B_270125/2048x2048/mask_filled_nuclei/count_pixels/"
    read_metrics_data(csv_file_path, encoding="ISO-8859-1" )
    
    # File path to npy file and return loaded images
    # npy_file_path = "/media/.../"
    # original_images = load_npy_images(npy_file_path)
    
    # Return combined dataframe
    # csv_file_path_comb = "/media/.../AugHoverData/all_pannuke_output/eval_func/pannuke_ensemble_all/00/results/"
    # combined_data = read_metrics_data_all_combined(csv_file_path_comb, "utf-8")

