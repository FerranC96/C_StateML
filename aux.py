
import pandas as pd
import numpy as np
import umap
import sys
import os

def read_marker_csv(input_dir):
    marker_files = [f for f in os.listdir(f"{input_dir}") if f.endswith(".csv")]
    if len(marker_files) != 1: #Sanity check
        sys.exit("ERROR: There should be ONE .csv file with the markers to use in the input folder!")
    else: #Get markers flagged for use
        marker_file = pd.read_csv(f"{input_dir}/{marker_files[0]}", header=None)
        selected_markers = marker_file.loc[marker_file[1] == "Y", [0]].values.tolist()
        selected_markers = [item for sublist in selected_markers for item in sublist]
    return selected_markers


def arcsinh_transf(cofactor, no_arc):
    #Select only the columns containing the markers (as they start with a number for the isotope)
    cols = [x for x in no_arc.columns if x[0].isdigit()]
    #Apply the arcsinh only to those columns (don't want to change time or any other)
    arc = no_arc.apply(lambda x: np.arcsinh(x/cofactor) if x.name in cols else x)
    # put back the 'file_origin' column to the arcsinh-transformed data
    if "file_origin" in  no_arc.columns:
        arc["file_origin"] = no_arc["file_origin"]
    # else:
    #     print ("(there was no concatenation prior to transforming)")
    return arc, cols

def downsample_data(no_arc, split_bycol, info_run, output_dir):
    downsampled_dframe = no_arc.copy()
    #Defiine downsampling size (N) per file:
    downsample_size = downsampled_dframe[split_bycol].value_counts().min() #at least N cells in all input files
    print ("Working with ", downsample_size, " cells per file_origin")
    #Group by file+origin and sample without replacement -> 
    # thus we can sample file for which len(file)=N without -tive consequences
    
    reduced_df = downsampled_dframe.groupby(split_bycol).apply(lambda x:
                                                    x.sample(downsample_size))
    
    #Create new file to store downsampling status for all cell IDs
    new_df = pd.DataFrame()
    new_df["Sample_ID-Cell_Index"] = no_arc["Sample_ID-Cell_Index"]
    new_df["In_donwsampled_file"] = new_df["Sample_ID-Cell_Index"].isin(
                                        reduced_df["Sample_ID-Cell_Index"])
    new_df.to_csv(f"{output_dir}/{info_run}_downsampled_IDs.csv", 
                    index = False)
    no_arc = no_arc[no_arc["Sample_ID-Cell_Index"].isin(reduced_df["Sample_ID-Cell_Index"])]
    return reduced_df