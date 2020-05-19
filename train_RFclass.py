import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from sklearn import metrics
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree.export import export_text
from joblib import dump
from aux import *
import seaborn as sns; sns.set()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~CONFIG~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
folder_name = "classRF"

if os.path.isdir(f"./output/{folder_name}") == False:
    os.makedirs(f"./output/{folder_name}")

    
input_dir = "../D_CommonDatasets/CRC-TME/Fibroblasts"
output_dir = f"./output/{folder_name}"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

info_run =  input("Write RF info run (using no spaces!): ")
os.makedirs(f"{output_dir}/{info_run}")

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

print ("Downsampling taking place. Check output folder for more info")
print (concat["cell-state_num"].value_counts())
dwns_concat = downsample_data(concat, "cell-state_num",f"{info_run}_downs_b4_RF", f"{output_dir}/{info_run}")
print (dwns_concat["cell-state_num"].value_counts())



processed_df = dwns_concat[cols].copy()
y = processed_df["cell-state_num"]
X = processed_df.drop("cell-state_num", axis=1)
#New X to drop @uninmportnat@ features/PTMs
# X = processed_df.drop(["cell-state_num","156Gd_pNF-kB p65","160Gd_pAMPKa","141Pr_pPDPK1","165Ho_Beta-Catenin_Active","153Eu_pCREB","147Sm_pBTK","170Er_pMEK1_2","148Nd_pSRC","168Er_pSMAD2_3","167Er_pERK1_2","163Dy_pP90RSK","157Gd_pMKK3_MKK6","154Sm_pSMAD1_5_9","166Er_pGSK3b","172Yb_pS6","155Gd_pAKT S473"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)


# clf = RandomForestRegressor(n_estimators=120, max_depth=None,
#                                 random_state=0, n_jobs=12)
clf = RandomForestClassifier(n_estimators=480, max_depth=None,
                                random_state=0, n_jobs=8) 

model_RFreg = clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

importances = model_RFreg.feature_importances_
std = np.std([tree.feature_importances_ for tree in model_RFreg.estimators_], axis=0)
indices = np.argsort(importances)[::-1]
# Print the feature ranking
print("Feature ranking:")
for f in range(X_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]), " -> " , X.columns[indices[f]])

print ("Score agains test data", model_RFreg.score(X_test, y_test))


plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.savefig(f"{output_dir}/{info_run}/{info_run}_feature_importances.png")

plt.figure()
plt.title("Prediction vs Real")
plt.scatter(y_test, predictions)
plt.plot(y_test, predictions, label=metrics.r2_score(y_test, predictions))
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.legend(loc='best')
plt.savefig(f"{output_dir}/{info_run}/{info_run}_pred_vs_real.png")

#Alternative to pickle that works better when storing large numpy arrays!
dump(clf, f"{output_dir}/{info_run}/{info_run}_RFcclass.joblib")
print("DEPRECATED SCRIPT. Used to generate the RF cycle classifier models")

#Get non-downs data
# processed_alldf = concat[cols].copy()

#Get diff set of data
processed_alldf = pd.DataFrame()
#Add counter to keep track of the number of files in input -> 
# -> cell ID will be a mix of these (Filenumber | filename.txt)
fcounter = 0
second_dir = "../D_CommonDatasets/CRC-TME/ALLcells"
filelist = [f for f in os.listdir(second_dir) if f.endswith(".txt")]
for file in filelist:
    name = file.split('.txt')[0]
    fcounter += 1
    df = pd.read_csv(f"{second_dir}/{file}", sep = '\t')
    df["file_origin"] = str(fcounter)+" | "+ file # add a new column of 'file_origin' that will be used to separate each file after umap calculation
    df["Sample_ID-Cell_Index"] = df["Cell_Index"].apply(
                                    lambda x: str(fcounter)+"-"+str(x)) #File+ID #This way the cell-index will be preserved after Cytobank upload
    # df["Cell_Index"] = df["Cell_Index"].apply(lambda x: str(fcounter)+"-"+str(x)) #File+ID
    processed_alldf = processed_alldf.append(df, ignore_index=True)

processed_alldf = processed_alldf[cols]


y_all = processed_alldf["cell-state_num"]
X_all = processed_alldf.drop("cell-state_num", axis=1)
# X_all = processed_alldf.drop(["cell-state_num","156Gd_pNF-kB p65","160Gd_pAMPKa","141Pr_pPDPK1","165Ho_Beta-Catenin_Active","153Eu_pCREB","147Sm_pBTK","170Er_pMEK1_2","148Nd_pSRC","168Er_pSMAD2_3","167Er_pERK1_2","163Dy_pP90RSK","157Gd_pMKK3_MKK6","154Sm_pSMAD1_5_9","166Er_pGSK3b","172Yb_pS6","155Gd_pAKT S473"], axis=1)
print ("Predictions on original non-downsampled data: ", model_RFreg.score(X_all, y_all))

predict_alldata = clf.predict(X_all)

# concat["Prediction"] = predict_alldata
# print ("Save predictions to predictions_all_concat.txt")
# concat.to_csv(f"{output_dir}/predictions_{info_run}_allconcat.txt", 
#                     index = False)
#Using the downsampled concatenated input as train (balanced states) the accurtacy score when testing all data goes down signficantly to just 50%

# print (metrics.r2_score(y_test, predictions))
print(metrics.classification_report(predictions,y_test))
print(metrics.classification_report(predict_alldata,y_all))


estimator = clf.estimators_[5]

from sklearn.tree import export_graphviz
# Export as dot file
export_graphviz(estimator, out_file=f"{output_dir}/{info_run}/{info_run}_tree.dot",
                feature_names=X_all.columns,
                class_names=["apoptosis","G0","G1","S","G2","M"],
                rounded = True, proportion = False, 
                precision = 2, filled = True)



mat_full = metrics.confusion_matrix(y_all, predict_alldata)
pd.DataFrame(mat_full).to_csv(f"{output_dir}/{info_run}/{info_run}_confusion_matrix_FULLdata_{model_RFreg.score(X_all, y_all)}.csv")

mat = metrics.confusion_matrix(y_test, predictions)
pd.DataFrame(mat).to_csv(f"{output_dir}/{info_run}/{info_run}_confusion_matrix_TESTdata_{model_RFreg.score(X_test, y_test)}.csv")

plt.figure()
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.savefig(f"{output_dir}/{info_run}/{info_run}_confusion_matrix.png")
plt.show()
##NORMALIZE MATRIX COUNT##
# mat_norm = mat.astype('float') / mat.sum(axis=1)[:, np.newaxis]
# plt.figure()
# sns.heatmap(mat_norm.T, square=True, annot=True, fmt='d', cbar=False)
# plt.xlabel('true label')
# plt.ylabel('predicted label')
# plt.show()






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
