import pandas as pd
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from aux import *

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~CONFIG~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
folder_name = "pre_classRF"

if os.path.isdir(f"./output/{folder_name}") == False:
    os.makedirs(f"./output/{folder_name}")
if os.path.isdir(f"./input/{folder_name}") == False:
    os.makedirs(f"./input/{folder_name}")
    sys.exit("ERROR: There is no input folder") 
    
input_dir = f"./input/{folder_name}"
output_dir = f"./output/{folder_name}"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

filelist = [f for f in os.listdir(input_dir) if f.endswith(".txt")]

print ("Input files:")
for i in filelist: 
    print (i)
    file_in = i

file = f"{input_dir}/{file_in}"
df_file = pd.read_csv(file, sep = '\t')
print (df_file.describe(include='all'))


df_arc = arcsinh_transf(5, df_file)[0]

state_group = df_arc.groupby("cell-state")
print (df_arc.groupby("cell-state").size())

for name, group in state_group:
    print(name)
    print(group)
    group.to_csv(f"{output_dir}/{name}_arctrans", index = False, sep = '\t')