
import os
import sys
from xmlrpc.client import Unmarshaller

import numpy as np
import pandas as pd


def readCellState(input_dir, ext=".txt" ,sep="\t"):
    # Function to read either a dataset annotated with cell state labels or
        # multiple files generated from the same experiment and where each file 
        # corresponds to a cell state gate and is named accordingly in the name
    
    dDataFrame = pd.DataFrame()
    filelist = [f for f in os.listdir(input_dir) if f.endswith(ext)]
    print ("Input files:")
    if len(filelist) ==1:
        print("(Only one)")
        print (filelist[0])
        df = pd.read_csv(f"{input_dir}/{filelist[0]}", sep = sep)
        df["Sample_ID-Cell_Index"] = df["Cell_Index"]
        if "cell-state_num" not in df.columns:
            sys.exit("ERROR: NO CELL STATE INFO (missing column called 'cell-state_num')")
        dDataFrame = pd.concat([dDataFrame, df], ignore_index=True)
    elif len(filelist)>1:
        print(f"({len(filelist)} files)")
        fcounter = 0
        for i in filelist: #TODO: Update with newer snippets from CyGNAL
            print (i)
            name = i.split(ext)[0]
            fcounter += 1
            df = pd.read_csv(f"{input_dir}/{i}", sep = sep)
            df["file_origin_RFcs"] = str(fcounter)+"_|_"+ name
            df["Sample_ID-Cell_Index"] = df["Cell_Index"].apply(
                                        lambda x: str(fcounter)+"-"+str(x)) #File+ID #This way the cell-index will be preserved after Cytobank upload
            if "cell-state_num" not in df.columns:
                print("Train data missing cell state information.",
                "Trying to label if info present in file name")
                if "apoptosis" in i.lower():
                    print(i.lower())
                    print(f"File {i} labelled as apoptosis")
                    df["cell-state"] = "apoptosis"
                    df["cell-state_num"] = 0
                elif "s-phase" in i.lower() or "s_phase" in i.lower():
                    print(i.lower())
                    print(f"File {i} labelled as s-phase")
                    df["cell-state"] = "s-phase"
                    df["cell-state_num"] = 3
                elif "m-phase" in i.lower() or "m_phase" in i.lower():
                    print(i.lower())
                    print(f"File {i} labelled as m-phase")
                    df["cell-state"] = "m-phase"
                    df["cell-state_num"] = 5
                elif "g0" in i.lower():
                    print(i.lower())
                    print(f"File {i} labelled as g0")
                    df["cell-state"] = "g0"
                    df["cell-state_num"] = 1
                elif "g1" in i.lower():
                    print(i.lower())
                    print(f"File {i} labelled as g1")
                    df["cell-state"] = "g1"
                    df["cell-state_num"] = 2
                elif "g2" in i.lower():
                    print(i.lower())
                    print(f"File {i} labelled as g2")
                    df["cell-state"] = "g2"
                    df["cell-state_num"] = 4
                else:
                    sys.exit(f"ERROR: File {i} could not be assigned to a cell state. Check your file names!")
            dDataFrame = pd.concat([dDataFrame, df], ignore_index=True)

    else:
        sys.exit(f"ERROR: There are no {ext} files in {input_dir}!")

    return filelist,dDataFrame


def readRFcs_data(labels_input, input_dir, ext=".csv", sep=","):
    concat_df = False
    dInput_list = []
    filelist = [f for f in os.listdir(input_dir) if f.endswith(ext)]
    if len(filelist)==0:
        sys.exit(f"ERROR: There are no {ext} files in {input_dir}!")
    else:
        print ("Input files:")
        for i in filelist:
            print(i)
            if labels_input == True:
                df = pd.read_csv(f"{input_dir}/{i}", sep = sep)
                if "cell-state_num" not in df.columns:
                    print("Trying to label with info present in file name")
                    if "apoptosis" in i.lower():
                        print(i.lower())
                        print(f"File {i} labelled as apoptosis")
                        df["cell-state_num"] = 0
                        concat_df = True
                    elif "s-phase" in i.lower() or "s_phase" in i.lower():
                        print(i.lower())
                        print(f"File {i} labelled as s-phase")
                        df["cell-state_num"] = 3
                        concat_df = True
                    elif "m-phase" in i.lower() or "m_phase" in i.lower():
                        print(i.lower())
                        print(f"File {i} labelled as m-phase")
                        df["cell-state_num"] = 5
                        concat_df = True
                    elif "g0" in i.lower():
                        print(i.lower())
                        print(f"File {i} labelled as g0")
                        df["cell-state_num"] = 1
                        concat_df = True
                    elif "g1" in i.lower():
                        print(i.lower())
                        print(f"File {i} labelled as g1")
                        df["cell-state_num"] = 2
                        concat_df = True
                    elif "g2" in i.lower():
                        print(i.lower())
                        print(f"File {i} labelled as g2")
                        df["cell-state_num"] = 4
                        concat_df = True
                    else:
                        print("WARNING!: Couldn't label input data, reverting to unlabelled mode")
                        labels_input = False # beware of semi-labelled data!!!!
                if concat_df == True and labels_input == True: #Auto labelling worked
                    #len(dInput_list) will be 1, containing a concatenated dataframe
                    try: #Concatenate new cell-state to existing [0]
                        dInput_list[0] = pd.concat([dInput_list[0], df], ignore_index=True)
                    except IndexError: #First pass should land here, where we will create empyt Df at [0]
                        dInput_list.append(pd.DataFrame())
                        dInput_list[0] = pd.concat([dInput_list[0], df], ignore_index=True)
                elif labels_input == True: #Label present from the beggining -> just append datasets to list
                    dInput_list.append(df)
            if labels_input == False: #Don't use elif since we want to evaluate the else case change above
                df = pd.read_csv(f"{input_dir}/{i}", sep = sep)
                dInput_list.append(df)
    
    return labels_input, dInput_list


def read_marker_csv(input_dir):
    marker_files = [f for f in os.listdir(f"{input_dir}") if f.endswith(".csv")]
    if len(marker_files) != 1: #Sanity check
        sys.exit("ERROR: There should be ONE .csv file with the markers to use in the input folder!")
    else: #Get markers flagged for use
        marker_file = pd.read_csv(f"{input_dir}/{marker_files[0]}", header=None)
        selected_markers = marker_file.loc[marker_file[1] == "Y", [0]].values.tolist()
        selected_markers = [item for sublist in selected_markers for item in sublist]
    return selected_markers


def translateAbMarkers(dataframe, marker_list):
    translation = {} #Translation layer needed since it's uncommon for names to be exact matches.
    unmmatched = []
    cold_storage = {}
    for i in dataframe.columns:
        if i in marker_list:
            print("Exact match")
            print (i)
            translation[i] = i
        else:
            for i2 in marker_list:
                try:
                    if i.split("_")[1] in i2.split("_")[1]:
                        if "".join(i.split("_")[1:3]) == "".join(i2.split("_")[1:3]):
                            #grabing [1:] only works if no version to Ab. Better to grab [1:3], as it should grab just Ab name and PTM site
                            print("Same marker, different channel/version")
                            print("from model", i2)
                            print("from input data", i)
                            translation[i] = i2
                        elif len(i2.split("_")) == 2: #Marker with only isotope and AbNAME
                            print("Marker with shorter name in input")
                            print("from model", i2)
                            print("from input data", i)
                            translation[i] = i2
                        elif len(i.split("_")) == 2: #Marker with only isotope and AbNAME
                            print("Marker with shorter name in model Ab list")
                            print("from model", i2)
                            print("from input data", i)
                            translation[i] = i2
                        elif "".join(i.split("_")[1]) == "".join(i2.split("_")[1]): #Fuzziest of matches
                            print("WARNING: Fuzzy match! Please manually check.\n",
                            "Marker will only be added to dictionary if it doesn't generate a duplicate")
                            print("from model", i2)
                            print("from input data", i)
                            cold_storage[i] = i2 #Avoid duplicates!
                        else:
                            print("ERROR: UNABLE TO PROPERLY MATCH!",i2, i)
                            unmmatched.append(i2)
                except:
                    pass
    for k,v in cold_storage.items(): #ENsure no duplicated due to fuzzy matching
        if v not in translation.values():
            translation[k] = v

    if len(translation) != len(marker_list):
        print("Translated markers",translation)
        print("Unmatched markers", unmmatched)
        print("Model markers", marker_list)
        for i in marker_list:
            if i not in translation.values():
                print(f"ERROR: Missing the following model feature in input data \n {i}")
                sys.exit(f"ERROR: Missing model features in input data!")
            #Something should happen here!
        #IN the future intead of sys.exit warn user and try to use the minimal 5 marker model.
    print("Translation dictionary: ", translation)
    dataframe = dataframe.rename(columns=translation)
    
    return dataframe


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
    #Create new file to store downsampling status for all cell IDs
    new_df = pd.DataFrame()
    os.makedirs(f"{output_dir}/{info_run}", exist_ok = True)
    #Update this so it relies on df Index instrad of the weird column!
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
