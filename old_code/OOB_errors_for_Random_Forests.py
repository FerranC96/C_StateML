import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from sklearn import metrics
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree.export import export_text
from aux import *
import seaborn as sns; sns.set()
from collections import OrderedDict

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~CONFIG~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
folder_name = "classRF"

if os.path.isdir(f"./output/{folder_name}") == False:
    os.makedirs(f"./output/{folder_name}")
if os.path.isdir(f"./input/{folder_name}") == False:
    os.makedirs(f"./input/{folder_name}")
    sys.exit("ERROR: There is no input folder") 
    
input_dir = f"./input/{folder_name}"
output_dir = f"./output/{folder_name}"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

info_run =  input("Write EMD info run (using no spaces!): ")

filelist = [f for f in os.listdir(input_dir) if f.endswith(".txt")]

print ("Input files:")
for i in filelist: 
    print (i)

cols = read_marker_csv(input_dir)
cols.append("cell-state_num") #Later on cell-state_num

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
    # df["Cell_Index"] = df["Cell_Index"].apply(lambda x: str(fcounter)+"-"+str(x)) #File+ID
    concat = concat.append(df, ignore_index=True)

# print("Concatenating...")
# concat.to_csv(f'{output_dir}/concat_{info_run}.txt', index = False, sep = '\t')
# print(f"Concatenated file saved as:\nconcat_{info_run}.txt")



#Downsampling section
if concat["file_origin"].value_counts().size > 1:
    print ("Downsampling taking place. Check output folder for more info")
    print (concat["file_origin"].value_counts())
    dwns_concat = downsample_data(concat, f"{info_run}_downs_b4_RF", output_dir)
    print (dwns_concat["file_origin"].value_counts())
else:
    print ("Only one input file detected; no downsampling")


processed_df = dwns_concat[cols].copy()
y = processed_df["cell-state_num"]
# X = processed_df.drop("cell-state_num", axis=1)
#New X to drop @uninmportnat@ features/PTMs
X = processed_df.drop(["cell-state_num","156Gd_pNF-kB p65","160Gd_pAMPKa","141Pr_pPDPK1","165Ho_Beta-Catenin_Active","153Eu_pCREB","147Sm_pBTK","170Er_pMEK1_2","148Nd_pSRC","168Er_pSMAD2_3","167Er_pERK1_2","163Dy_pP90RSK","157Gd_pMKK3_MKK6","154Sm_pSMAD1_5_9","166Er_pGSK3b","172Yb_pS6","155Gd_pAKT S473"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.02)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)
ensemble_clfs = [ ("RF, auto/sqrt", RandomForestClassifier(n_estimators=480, max_depth=None,
                                random_state=0, warm_start=True, oob_score=True))] 

min_estimators = 100
max_estimators = 600

error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

for label, clf in ensemble_clfs:
    for i in range(min_estimators, max_estimators + 20):
        print ("Start oob")
        clf.set_params(n_estimators=i)
        clf.fit(X_train, y_train)

        # Record the OOB error for each `n_estimators=i` setting.
        oob_error = 1 - clf.oob_score_
        error_rate["RF, auto/sqrt"].append((i, oob_error))
        print ("Finish OOB step")

for label, clf_err in error_rate.items():
    xs, ys = zip(*clf_err)
    plt.plot(xs, ys, label=label)

plt.xlim(min_estimators, max_estimators)
plt.xlabel("n_estimators")
plt.ylabel("OOB error rate")
plt.legend(loc="upper right")
plt.show()