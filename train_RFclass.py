import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
# import magic
import tasklogger
from sklearn import metrics
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import export_text
from joblib import dump
from aux import *
import seaborn as sns; sns.set()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~CONFIG~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
folder_name = "classRF"

if os.path.isdir(f"./output/{folder_name}") == False:
    os.makedirs(f"./output/{folder_name}")

    
input_dir = "../D_CyTOF/Data4CellStateCLASS/PDO21_EGF-Titration/PDO21_Untreated_rep1" #Contains normalised data(no need to splitby)
second_dir = "../D_CyTOF/Data4CellStateCLASS/PDO21_EGF-Titration/PDO21_10ng"
# second_dir = "../D_CommonDatasets/C_Fig2/States" #Shoudl eb same as above unless testing
output_dir = f"./output/{folder_name}"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

info_run =  input("Write RF info run (using no spaces!): ")

if os.path.isdir(f"{output_dir}/TRAINING_{info_run}") == True:
    print("THIS INFO RUN HAS ALREADY BEEN USED. BE aware of overwritting data")
else:
    os.makedirs(f"{output_dir}/TRAINING_{info_run}")

filelist = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
if len(filelist) == 0:
    sys.exit(f"ERROR: There are no .txt files in {input_dir}!")
#Check the files found in the directory:
print ("Input files in input_dir:")
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

# print("Concatenating...")
# concat.to_csv(f'{output_dir}/concat_{info_run}.txt', index = False, sep = '\t')
# print(f"Concatenated file saved as:\nconcat_{info_run}.txt")

#Sanity check for presence of cell-state columns:
if "cell-state" not in concat.columns:
    sys.exit("No cell state INFO!!!")
    

#Downsampling section

print ("Downsampling taking place. Check output folder for more info")
print (concat["cell-state_num"].value_counts())
dwns_concat = downsample_data(concat, "cell-state_num",f"{info_run}_downs_b4_RF", f"{output_dir}/TRAINING_{info_run}")
print (dwns_concat["cell-state_num"].value_counts())

# print(dwns_concat.head)

###################################################################

processed_df = dwns_concat[cols].copy()
# print(processed_df)

y = processed_df["cell-state_num"]
X = processed_df.drop("cell-state_num", axis=1)

#Denoise input training data with MAGIC
# print(X.head())
# with tasklogger.log_task("monocore"):
#     magic_op = magic.MAGIC(knn=5, n_jobs=-2)
#     X_denoised = magic_op.fit_transform(X)
# print(X_denoised)
# X = X_denoised

#New X to drop @uninmportnat@ features/PTMs
# X = processed_df.drop(["cell-state_num","156Gd_pNF-kB p65","160Gd_pAMPKa","141Pr_pPDPK1","165Ho_Beta-Catenin_Active","153Eu_pCREB","147Sm_pBTK","170Er_pMEK1_2","148Nd_pSRC","168Er_pSMAD2_3","167Er_pERK1_2","163Dy_pP90RSK","157Gd_pMKK3_MKK6","154Sm_pSMAD1_5_9","166Er_pGSK3b","172Yb_pS6","155Gd_pAKT S473"], axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.05)
print (X_train.shape, y_train.shape)
print (X_valid.shape, y_valid.shape)


# clf = RandomForestRegressor(n_estimators=120, max_depth=None,
#                                 random_state=0, n_jobs=12)
clf = RandomForestClassifier(n_estimators=420, max_depth=None,
                                random_state=0, n_jobs=8) 

model_RFreg = clf.fit(X_train, y_train)

predictions = clf.predict(X_valid)

importances = model_RFreg.feature_importances_
std = np.std([tree.feature_importances_ for tree in model_RFreg.estimators_], axis=0)
indices = np.argsort(importances)[::-1]
# Print the feature ranking
print("Feature ranking:")
for f in range(X_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]), " -> " , X.columns[indices[f]])


plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation="vertical")
plt.xlim([-1, X.shape[1]])
plt.savefig(f"{output_dir}/TRAINING_{info_run}/{info_run}_feature_importances.png", bbox_inches = "tight")

# plt.figure()
# plt.title("Prediction vs Real")
# plt.scatter(y_valid, predictions)
# plt.plot(y_valid, predictions, label=metrics.r2_score(y_valid, predictions))
# plt.xlabel("True Values")
# plt.ylabel("Predictions")
# plt.legend(loc='best')
# plt.savefig(f"{output_dir}/TRAINING_{info_run}/{info_run}_pred_vs_real.png")

#Alternative to pickle that works better when storing large numpy arrays!
dump(clf, f"{output_dir}/TRAINING_{info_run}/{info_run}_RFcclass.joblib")
print("DEPRECATED SCRIPT. Used to generate the RF cycle classifier models") 

#Get non-downs data
# processed_alldf = concat[cols].copy()


##############################################
#TEST DATA

filelist2 = [f for f in os.listdir(second_dir) if f.endswith(".txt")]
if len(filelist2) == 0:
    sys.exit(f"ERROR: There are no .txt files in {second_dir}!")
#Check the files found in the directory:
print ("Input files in second_dir:")
for i in filelist2:
    print (i)

test_df = pd.DataFrame()
#Add counter to keep track of the number of files in input -> 
# -> cell ID will be a mix of these (Filenumber | filename.txt)
fcounter = 0
for file in filelist2:
    name = file.split('.txt')[0]
    fcounter += 1
    df = pd.read_csv(f"{second_dir}/{file}", sep = '\t')
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

    test_df = test_df.append(df, ignore_index=True)

test_df = test_df[cols] #Must use shared collumns bwten train and test data

y_test = test_df["cell-state_num"]
X_test = test_df.drop("cell-state_num", axis=1)

#Denoise input testing data with MAGIC
# print(X_all.head())
# with tasklogger.log_task("monocore_test"):
#     magic_op = magic.MAGIC(knn=5, n_jobs=-2)
#     X_alldenoised = magic_op.fit_transform(X_all)
# print(X_alldenoised)
# X_all = X_alldenoised

# X_all = processed_alldf.drop(["cell-state_num","156Gd_pNF-kB p65","160Gd_pAMPKa","141Pr_pPDPK1","165Ho_Beta-Catenin_Active","153Eu_pCREB","147Sm_pBTK","170Er_pMEK1_2","148Nd_pSRC","168Er_pSMAD2_3","167Er_pERK1_2","163Dy_pP90RSK","157Gd_pMKK3_MKK6","154Sm_pSMAD1_5_9","166Er_pGSK3b","172Yb_pS6","155Gd_pAKT S473"], axis=1)

predict_test = clf.predict(X_test)

# concat["Prediction"] = predict_alldata
# print ("Save predictions to predictions_all_concat.txt")
# concat.to_csv(f"{output_dir}/predictions_{info_run}_allconcat.txt", 
#                     index = False)
#Using the downsampled concatenated input as train (balanced states) the accurtacy score when testing all data goes down signficantly to just 50%

# print (metrics.r2_score(y_test, predictions))
print ("Score agains validation data", model_RFreg.score(X_valid, y_valid))
print(metrics.classification_report(predictions,y_valid))
json.dump(metrics.classification_report(predictions,y_valid, output_dict=True), open(f"{output_dir}/TRAINING_{info_run}/{info_run}_ClassReport_VALIDdata_{model_RFreg.score(X_valid, y_valid)}.json", "w"))
print ("Predictions on test data: ", model_RFreg.score(X_test, y_test))
print(metrics.classification_report(predict_test,y_test))
json.dump(metrics.classification_report(predict_test,y_test, output_dict=True), open(f"{output_dir}/TRAINING_{info_run}/{info_run}_ClassReport_TESTdata_{model_RFreg.score(X_test, y_test)}.json", "w"))

#Log loss:
logloss_score = metrics.log_loss(y_test, clf.predict_proba(X_test))
print("Log loss score is:", logloss_score)


#Confussion matrices

mat_test = metrics.confusion_matrix(y_test, predict_test)
pd.DataFrame(mat_test).to_csv(f"{output_dir}/TRAINING_{info_run}/{info_run}_confusion_matrix_TESTdata_{model_RFreg.score(X_test, y_test)}.csv")

mat = metrics.confusion_matrix(y_valid, predictions)
pd.DataFrame(mat).to_csv(f"{output_dir}/TRAINING_{info_run}/{info_run}_confusion_matrix_VALIDATIONdata_{model_RFreg.score(X_valid, y_valid)}.csv")

plt.figure()
sns.heatmap(mat_test, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.savefig(f"{output_dir}/TRAINING_{info_run}/{info_run}_confusionmatrix_TEST.png")
plt.show()

plt.figure()
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.savefig(f"{output_dir}/TRAINING_{info_run}/{info_run}_confusionmatrix_VALIDATION.png")
plt.show()

##NORMALIZE MATRIX COUNT##
# mat_norm = mat.astype('float') / mat.sum(axis=1)[:, np.newaxis]
# plt.figure()
# sns.heatmap(mat_norm.T, square=True, annot=True, fmt='d', cbar=False)
# plt.xlabel('true label')
# plt.ylabel('predicted label')
# plt.show()


#Tree from the RF

estimator = clf.estimators_[5]

from sklearn.tree import export_graphviz
# Export as dot file
export_graphviz(estimator, out_file=f"{output_dir}/TRAINING_{info_run}/{info_run}_tree.dot",
                feature_names=X_test.columns,
                class_names=["apoptosis","G0","G1","S","G2","M"],
                rounded = True, proportion = False, 
                precision = 2, filled = True)


# # #Testing ROC plots #ONLY WITH BINARY CLASSIFICATION!!!
# y_pred_0 = clf.predict_proba(X_test)[:,0]
# try:
#     fpr_rf, tpr_rf, _ = metrics.roc_curve(y_test, y_pred_0)
# except:
#     print ("ROC couldn't be calculated")
# # plt.figure()
# # plt.plot(fpr_rf, tpr_rf, label='RF')
# # plt.xlabel('False positive rate')
# # plt.ylabel('True positive rate')
# # plt.title('ROC curve for Apoptosis')
# # plt.legend(loc='best')
# # plt.show()


# kf = KFold(n_splits=5) # Define the split - into 2 folds
# kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator
# print (kf)
# for train_index, test_index in kf.split(X):
#     print ("TRAIN:", train_index, "TEST:", test_index)
# #  X_train, X_test = X[train_index], X[test_index]
# #  y_train, y_test = y[train_index], y[test_index]('TRAIN:', array([2, 3]), 'TEST:', array([0, 1]))
# # ('TRAIN:', array([0, 1]), 'TEST:', array([2, 3]))


    # f_reduced.to_csv(f"{output_dir}/{i}", index = False, sep = '\t') s
        # index = False to be compatible with Cytobank    
