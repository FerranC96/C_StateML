import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
# import magic
import tasklogger
from sklearn import metrics
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_text
from joblib import dump
from aux import readCellState, read_marker_csv, downsample_data, translateAbMarkers, yes_or_NO
import seaborn as sns; sns.set()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~I/O~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
folder_name = "trainRF"

if os.path.isdir(f"./output/{folder_name}") == False:
    os.makedirs(f"./output/{folder_name}")
#INput dir to get train and validation data (downsampled). Second_dir for test2
train_dir = "../D_CyTOF/Data4CellStateCLASS/PDO21_EGF-Titration/PDO21_ALL_rep2" #Change as appropiate
test2_dir = "../D_CyTOF/CRC-TME/Epithelial-Cells" #Test against colon organoids from natmethods Fig5

output_dir = f"./output/{folder_name}"

info_run =  input("Write info run (using no spaces!): ")
if len(info_run) == 0:
    print("No info run given. Saving results in UNNAMED")
    info_run = "UNNAMED"

if os.path.isdir(f"{output_dir}/{info_run}") == False:
    os.makedirs(f"{output_dir}/{info_run}")
else:
    if info_run !="UNNAMED":
        sys.exit("ERROR: You already used this name for a previous run. \nUse a different name!")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


#~~~~~~~~~~~~~~~~~~~~~Prepare TRAIN and validation data~~~~~~~~~~~~~~~~~~~~#
print("Train directory: ", train_dir)
#Read and cellstate labelled data, can use file name info to create cs label
filelist, dTrain = readCellState(train_dir, ext=".txt", sep="\t")

#Define features to build model:
cols = read_marker_csv(train_dir)
#['89Y_pHH3', '127I_IdU', '142Nd_cCaspase 3', '150Nd_pRB', '176Yb_Cyclin B1']
#Ensure AB names have no spaces!:
cols = [i.replace(" ","_") for i in cols]
cols.append("cell-state_num") #Add cell state to cols to keep

# #DEPRECATED
# concat = label_concatenate(filelist, input_dir)

# save_concat = yes_or_NO("Save input df concat as one with cell state info columns?")
# if save_concat:
#     print("Concatenating...")
#     concat.to_csv(f"{output_dir}/TRAINING_{info_run}/{info_run}_concatIN.csv", 
#                     index = False, sep = '\t')
#     print(f"Concatenated file saved as:\n{info_run}_concatIN.csv")


#Balance classes by donwsampling:
if (dTrain["cell-state_num"].value_counts()[0]==dTrain["cell-state_num"].value_counts()).all():
    print("JACK-POT! Downsampling to balance classes not required")
    dTrain_dwnsBalanced = dTrain
else:
    print ("Downsampling taking place. Check output folder for more info")
    print (dTrain["cell-state_num"].value_counts())
    dTrain_dwnsBalanced = downsample_data(dTrain, f"dwnsBalanced_b4_RF", 
                f"{output_dir}/{info_run}", 
                split_bycol="cell-state_num")

print (dTrain_dwnsBalanced["cell-state_num"].value_counts())

#Translation section: Not strcilty needed here as panel markers should have been 
# generated with this datasets

dTrain_dwnsBalanced.columns = [i.replace(" ","_") for i in dTrain_dwnsBalanced.columns]

dTrain_dwnsBalanced = translateAbMarkers(dTrain_dwnsBalanced, cols)

#Subset to cols
# dTrain_dwnsBalanced = dTrain_dwnsBalanced.drop(columns=np.setdiff1d(dTrain_dwnsBalanced.columns, cols))
dTrain_dwnsBalanced = dTrain_dwnsBalanced[cols] #Much simpler than above!


#Section to check if data has been arcsinh transformed:
if dTrain_dwnsBalanced.max().astype("float64").max() > 12:
    print("WARNING! IS YOUR DATA NORMALISED? \n CyTOF data is generally normalised using arcsinh(a=5), and it seems like your data might not have been normalised.")
    #Eventually have some code to prompt the user to normalise it here if wanted
    stop_script = yes_or_NO("Do you want to exit the script?")
    if stop_script:
        #Do something if yes/true
        #Something being to exit the run
        sys.exit("Exiting the script")


#~~~~~~~~~~~~~~~~~~~~~TRAIN Random Forest~~~~~~~~~~~~~~~~~~~~#
Y = dTrain_dwnsBalanced["cell-state_num"]
X = dTrain_dwnsBalanced.drop("cell-state_num", axis=1)

#Denoise input training data with MAGIC
# print(X.head())
# with tasklogger.log_task("ALLbut2"):
#     magic_op = magic.MAGIC(knn=5, n_jobs=-2)
#     X_denoised = magic_op.fit_transform(X)
# print(X_denoised)
# X = X_denoised

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05)
print (X_train.shape, Y_train.shape)
print (X_test.shape, Y_test.shape)
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.12)
print (X_train.shape, Y_train.shape)
print (X_valid.shape, Y_valid.shape)


mRF = RandomForestClassifier(n_estimators=420, 
                            max_features="sqrt", 
                            max_depth=None, min_samples_split=2,
                            random_state=12, oob_score=True,
                            n_jobs=-3)  

mRF.fit(X_train, Y_train)
mRFcal = CalibratedClassifierCV(mRF, method="sigmoid", cv="prefit")
mRFcal.fit(X_valid, Y_valid)

mRF_probs = mRF.predict_proba(X_test)
mRFcal_probs = mRFcal.predict_proba(X_test)

mRF_preds = mRF.predict(X_test)
mRFcal_preds = mRFcal.predict(X_test)
print("(Are the) base and calibrated model predictions equal(?)", 
        np.array_equal(mRF_preds, mRFcal_preds))

mRF_cvscores = cross_val_score(mRF, X, Y)
print(f"CV score of base model: {mRF_cvscores.mean()} (+/- {mRF_cvscores.std()*2})")

# print(metrics.classification_report(mRF_preds,Y_test))
# print(metrics.classification_report(mRFcal_preds,Y_test))



#OLD
# #Standard RF on whole train
# model_RFreg = clf.fit(X_train, y_train)

# #Use train for RF and then validation for sigmoid callibration
# # model_RFreg = clf.fit(X_train, y_train)
# clf_probs = clf.predict_proba(X_valid)
# sig_clf = CalibratedClassifierCV(clf, method="sigmoid", cv="prefit")
# sigmodel_RFreg = sig_clf.fit(X_valid, y_valid)


# #Predictions from model and CV scores
# predictions = clf.predict(X_valid)

# print("Cross validation scores (mean and 95% CI):")
# cv_scores = cross_val_score(model_RFreg, X, y)
# print(f"Accuracy: {cv_scores.mean()} (+/- {cv_scores.std()*2})")





#~~~~~~~~~~~~~~~~~~~~~TRAIN model features~~~~~~~~~~~~~~~~~~~~#
#Instead of accessing mRF and mRFcal separately, we can acces mRF within mRF cal
print(mRFcal.base_estimator)
print(mRFcal)
mRF_feats = mRFcal.base_estimator.feature_names_in_
mRFcal_feats = mRFcal.feature_names_in_

mRF_import = mRFcal.base_estimator.feature_importances_ 
importstd = np.std([tree.feature_importances_ for tree in mRFcal.base_estimator.estimators_], axis=0)
feat_index = np.argsort(mRF_import)[::-1]
# Print the feature ranking
print("Feature importance ranking:")
for i in feat_index:
    print(f"Feature {i} ({mRFcal_feats[i]}) -> {mRF_import[i]}")
# for f in range(X_train.shape[1]):
#     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]), " -> " , X.columns[indices[f]])

plt.figure()
plt.title("Feature importance plot")
plt.bar(range(len(mRFcal_feats)), mRF_import[feat_index],
        color="r", yerr=importstd[feat_index], align="center")
plt.xticks(range(len(mRFcal_feats)), mRFcal_feats[feat_index], rotation="vertical")
plt.xlim([-1, len(mRFcal_feats)])
plt.savefig(f"{output_dir}/{info_run}/feature_importances.png", bbox_inches = "tight")

# plt.figure()
# plt.title("Prediction vs Real")
# plt.scatter(y_valid, predictions)
# plt.plot(y_valid, predictions, label=metrics.r2_score(y_valid, predictions))
# plt.xlabel("True Values")
# plt.ylabel("Predictions")
# plt.legend(loc='best')
# plt.savefig(f"{output_dir}/TRAINING_{info_run}/{info_run}_pred_vs_real.png")

#Alternative to pickle that works better when storing large numpy arrays!
dump(mRFcal, f"{output_dir}/{info_run}/{info_run}_RFcclass.joblib")

#Get non-downs data
# processed_alldf = concat[cols].copy()


#~~~~~~~~~~~~~~~~~~~~~Prepare TEST data~~~~~~~~~~~~~~~~~~~~#
#UPDATE BELOW WITH NEW FORMAT AND FUNCTIONS:
print("Test2 directory: ", test2_dir)
#Generate df, load in files either as one with labels or mutliple with labels in the file namme
#Translate columns, subset to just them.
#no downsampling
#Check for arcsinh

dTest = readCellState(test2_dir, ext=".txt", sep="\t")[1]

dTest.columns = [i.replace(" ","_") for i in dTest.columns]

#Translate:
dTest = translateAbMarkers(dTest, cols)

#Subset to cols
dTest = dTest[cols]

#Check if data has been arcsinh transformed:
if dTest.max().max() > 12:
    print("WARNING! IS YOUR DATA NORMALISED? \n CyTOF data is generally normalised using arcsinh(a=5), and it seems like your data might not have been normalised.")
    #Eventually have some code to prompt the user to normalise it here if wanted

#~~~~~~~~~~~~~~~~~~~~~TEST Random Forest~~~~~~~~~~~~~~~~~~~~#

Y_test2 = dTest["cell-state_num"]
X_test2 = dTest.drop("cell-state_num", axis=1)

mRF_preds_test2 = mRF.predict(X_test2)
mRFcal_preds_test2 = mRFcal.predict(X_test2)
print("(Are the) base and calibrated model predictions equal(?)", 
        np.array_equal(mRF_preds_test2, mRFcal_preds_test2))

# print("Cross validation scores for test2 data (mean and 95% CI):")
# mRF_cvscores_test2 = cross_val_score(mRF, X_test2, Y_test2)
# print(f"CV score of base model: {mRF_cvscores_test2.mean()} (+/- {mRF_cvscores_test2.std()*2})")

# print(metrics.classification_report(mRF_preds_test2,Y_test2))
# print(metrics.classification_report(mRFcal_preds_test2,Y_test2))


#~~~~~~~~~~~~~~~~~~~~~TEST/validation metrics and plots~~~~~~~~~~~~~~~~~~~~#


print ("Performance against test data", mRFcal.score(X_test, Y_test))
print(metrics.classification_report(mRFcal_preds,Y_test))
json.dump(metrics.classification_report(mRFcal_preds,Y_test, output_dict=True), 
            open(f"{output_dir}/{info_run}/ClassReport_VALIDdata_{mRFcal.score(X_test, Y_test)}.json", "w"))

print ("Performance against test2 data", mRFcal.score(X_test2, Y_test2))
print(metrics.classification_report(mRFcal_preds_test2,Y_test2))
json.dump(metrics.classification_report(mRFcal_preds_test2,Y_test2, output_dict=True), 
            open(f"{output_dir}/{info_run}/ClassReport_TESTdata_{mRFcal.score(X_test2, Y_test2)}.json", "w"))

#Log loss:
logloss_score = metrics.log_loss(Y_test2, mRFcal.predict_proba(X_test2))
print("Log loss score is:", logloss_score)

#Confussion matrices
mat = metrics.confusion_matrix(Y_test, mRFcal_preds)
pd.DataFrame(mat).to_csv(f"{output_dir}/{info_run}/confusion_matrix_TEST1data_{mRFcal.score(X_test, Y_test)}.csv")
mat_test = metrics.confusion_matrix(Y_test2, mRFcal_preds_test2)
pd.DataFrame(mat_test).to_csv(f"{output_dir}/{info_run}/confusion_matrix_TESTdata_{mRFcal.score(X_test2, Y_test2)}.csv")

plt.figure()
sns.heatmap(mat_test, square=True, annot=True, fmt="d", cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.savefig(f"{output_dir}/{info_run}/confusionmatrix_TEST2.png")
plt.show()

##NORMALIZE MATRIX COUNT##
# mat_norm = mat.astype('float') / mat.sum(axis=1)[:, np.newaxis]
# plt.figure()
# sns.heatmap(mat_norm.T, square=True, annot=True, fmt='d', cbar=False)
# plt.xlabel('true label')
# plt.ylabel('predicted label')
# plt.show()


#Tree from the RF

estimator = mRFcal.base_estimator.estimators_[12]

from sklearn.tree import export_graphviz
# Export as dot file
export_graphviz(estimator, out_file=f"{output_dir}/{info_run}/Sample_tree.dot",
                feature_names=mRFcal_feats,
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


