import pyvips
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path

from pixel_count import count_black_pixels
from read_csv_files import read_metrics_data

def statistics(mask_path: str | Path, csv_pixel_path: str | Path, csv_nuclei_number_path: str | Path, save_stats_path: str | Path):
    
    # Number of black pixels in mask represents number of pixels that are tissue in the image
    nbr_tissue_pixels_in_mask = count_black_pixels(mask_path)
    
    # Total number of pixels for each cell category found in wsi
    tot_pixels_df = read_metrics_data(csv_pixel_path, encoding="ISO-8859-1")
    # Total number of nuclei for each cell category found in wsi
    tot_nuclei_df = read_metrics_data(csv_nuclei_number_path, encoding="ISO-8859-1")

    tot_pixels_immune_cells = tot_pixels_df["neutrophil"].iloc[0] + tot_pixels_df["lymphocyte"].iloc[0] + tot_pixels_df["plasma"].iloc[0] + tot_pixels_df["eosinophil"].iloc[0]
    print("---------------------------------------------------------------------------------------")
    print(f"Total number of pixels that are immune cells in wsi: {tot_pixels_immune_cells}")
    print("---------------------------------------------------------------------------------------")
    tot_nuclei_immune_cells = tot_nuclei_df["neutrophil"].iloc[0] + tot_nuclei_df["lymphocyte"].iloc[0] + tot_nuclei_df["plasma"].iloc[0] + tot_nuclei_df["eosinophil"].iloc[0]
    print(f"Total number of immune cells in wsi: {tot_nuclei_immune_cells}")
    print("---------------------------------------------------------------------------------------")
    tot_nuclei_wsi = tot_nuclei_df["neutrophil"].iloc[0] + tot_nuclei_df["lymphocyte"].iloc[0] + tot_nuclei_df["plasma"].iloc[0] + tot_nuclei_df["eosinophil"].iloc[0] + tot_nuclei_df["connective"].iloc[0] + tot_nuclei_df["epithelial"].iloc[0]
    print(f"Total number of nuclei in wsi: {tot_nuclei_wsi}")
    print("---------------------------------------------------------------------------------------")
    percent_pixels_immune_cells = (tot_pixels_immune_cells/nbr_tissue_pixels_in_mask) * 100
    print(f"% of pixels categorized as immune cells in tissue region of wsi: {percent_pixels_immune_cells:.4f}")
    print("---------------------------------------------------------------------------------------")
    percent_nuclei_immune_cells = (tot_nuclei_immune_cells/tot_nuclei_wsi) * 100
    print(f"% of nuclei detected in wsi categorized as immune cells: {percent_nuclei_immune_cells:.4f}")
    print("---------------------------------------------------------------------------------------")
    # add stats mm2 
    pd.set_option('display.max_rows', None)     # Show all rows
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', 1000)        # Adjust the width of the display
    info_df = pd.DataFrame({"tot_pixels_immune_cells": tot_pixels_immune_cells,
                            "tot_nuclei_immune_cells":tot_nuclei_immune_cells,
                            "tot_nuclei_wsi": tot_nuclei_wsi,
                            "percent_pixels_immune_cells":percent_pixels_immune_cells,
                           "percent_nuclei_immune_cells": percent_nuclei_immune_cells}, index=[0])
    info_df.to_csv(save_stats_path, index=False)

if __name__ == "__main__":
    mask_path = "/media/jenny/Expansion/MM_HE_masks/HE_MM179_2D_290125_20x_BF_01/HE_MM179_2D_290125_20x_BF_01_full_mask.png"
    csv_nuclei_number_path = "/media/jenny/Expansion/MM_HE_results/HE_MM179_2D_290125_20x_BF_01/mask/counts/"
    csv_pixel_path = "/media/jenny/Expansion/MM_HE_results/HE_MM179_2D_290125_20x_BF_01/mask/count_pixels/"
    # Output directory
    save_stats_path = Path("/media/jenny/Expansion/MM_HE_results/HE_MM179_2D_290125_20x_BF_01/mask/stats/")
    if not save_stats_path.exists(): 
        save_stats_path.mkdir(parents=True)
        print(f"Directory {save_stats_path} was created")
    save_stats_path = os.path.join(save_stats_path, f"stats.csv")
    
    statistics(mask_path, csv_pixel_path, csv_nuclei_number_path, save_stats_path)