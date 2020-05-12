#Not crucial since the main script already splits by state. 
# Use to manipulate raw data and generate relevant subsets
import pandas as pd
import os
import sys
from aux import *

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~CONFIG~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
folder_name = "pre_classRF"

if os.path.isdir(f"./output/{folder_name}") == False:
    os.makedirs(f"./output/{folder_name}")
    
input_dir = "../D_CommonDatasets/CRC-TME/Epithelial-Cells"
output_dir = f"./output/{folder_name}"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

filelist = [f for f in os.listdir(input_dir) if f.endswith(".txt")]

print ("Input files:")
for i in filelist: 
    print (i)
    file_in = i

    file = f"{input_dir}/{file_in}"
    df_arc = pd.read_csv(file, sep = '\t')
    print (df_arc.describe(include='all'))

    splitby_group = df_arc.groupby("cell-state")
    print (df_arc.groupby("cell-state").size())

    for name, group in splitby_group:
        print(name)
        print(group)
        group.to_csv(f"{output_dir}/{i}_subset_{name}.txt", index = False, sep = '\t')