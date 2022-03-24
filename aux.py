
import os
import sys

import numpy as np
import pandas as pd
import umap


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

#Downsample dataframe by column and save to file which IDs were removed
#UPDATED IN 2022 to drop unused cell0state_num level after groupby
def downsample_data(no_arc, info_run, output_dir, 
                    split_bycol="file_identifier"): 
    downsampled_dframe = no_arc.copy()
    #Defiine downsampling size (N) per file: at least N cells in all input files
    downsample_size = downsampled_dframe[split_bycol].value_counts().min() 
    print ("Working with ", downsample_size, " cells per split")
    #Group by file+origin and sample without replacement -> 
    # thus we can sample file for which len(file)=N without -tive consequences 
    reduced_df = downsampled_dframe.groupby(split_bycol, 
                                            as_index=False).apply(lambda x:
                                                    x.sample(downsample_size)
                                                                ).droplevel(0)
    # reduced_df['new-cell-index'] = list(range(len(reduced_df.index)))
    # reduced_df['post_downsample-cell_index'] = reduced_df.index
    
    #Create new file to store downsampling status for all cell IDs
    new_df = pd.DataFrame()
    os.makedirs(f'{output_dir}/{info_run}', exist_ok = True)
    new_df["Sample_ID-Cell_Index"] = no_arc["Sample_ID-Cell_Index"]
    new_df["In_donwsampled_file"] = new_df["Sample_ID-Cell_Index"].isin(
                                    reduced_df["Sample_ID-Cell_Index"])
    new_df.to_csv(f"{output_dir}/{info_run}/{info_run}_downsampled_IDs.csv", 
                    index = False)
    no_arc = no_arc[no_arc["Sample_ID-Cell_Index"].isin(
                reduced_df["Sample_ID-Cell_Index"])]
    return reduced_df 

def label_concatenate(filelist, input_dir):
    concat = pd.DataFrame()
    #Add counter to keep track of the number of files in input -> 
    # -> cell ID will be a mix of these (Filenumber | filename.txt)
    fcounter = 0
    for file in filelist:
        name = file.split('.txt')[0]
        fcounter += 1
        df = pd.read_csv(f"{input_dir}/{file}", sep = '\t')
        df["file_origin"] = str(fcounter)+" | "+ file # add a new column of 'file_origin' that will be used to separate each file after umap calculation
        df["Sample_ID-Cell_Index"] = df["Cell_Index"].apply(
                                        lambda x: str(fcounter)+"-"+str(x)) #File+ID #This way the cell-index will be preserved after Cytobank upload
        #ADD CELLstate info when files are named XXXX.CellState_Phase.txt. Naming from CygNALs 6-c.py
        if "Apoptosis.txt" in file:
            df["cell-state"] = "apoptosis"
            df["cell-state_num"] = "0"
        elif "G0.txt" in file:
            df["cell-state"] = "g0"
            df["cell-state_num"] = "1"
        elif "G1.txt" in file:
            df["cell-state"] = "g1"
            df["cell-state_num"] = "2"
        elif "S_Phase.txt" in file:
            df["cell-state"] = "s-phase"
            df["cell-state_num"] = "3"
        elif "G2.txt" in file:
            df["cell-state"] = "g2"
            df["cell-state_num"] = "4"
        elif "M_Phase.txt" in file:
            df["cell-state"] = "m-phase"
            df["cell-state_num"] = "5"
        else:
            print(f"File {file} could not be assigned to a cell state. Check your file names!")

        concat = concat.append(df, ignore_index=True)
    return concat




#Simple yes or no input function (default NO)
def yes_or_NO(question, default="NO"):
    if default.lower() == "no":
        while True:
            reply = str(input(question+' (y/[N]): ')).lower().strip()
            if reply[:1] == 'y':
                return True
            elif reply[:1] == 'n':
                return False
            elif reply[:1] == "":
                return False
            else:
                print ("Please answer Y or N")
    elif default.lower() == "yes":
        while True:
            reply = str(input(question+' ([Y]/n): ')).lower().strip()
            if reply[:1] == 'y':
                return True
            elif reply[:1] == 'n':
                return False
            elif reply[:1] == "":
                return True
            else:
                print ("Please answer Y or N")

########################################################################
