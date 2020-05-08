import pandas as pd
import os
import sys
from aux import *

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~CONFIG~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
folder_name = "pre_classRF"

if os.path.isdir(f"./output/{folder_name}") == False:
    os.makedirs(f"./output/{folder_name}")
    
input_dir = "../D_CommonDatasets/C_Fig4Time"
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


state_group = df_arc.groupby("Day")
print (df_arc.groupby("Day").size())

for name, group in state_group:
    print(name)
    print(group)
    group.to_csv(f"{output_dir}/{name}_subset.txt", index = False, sep = '\t')