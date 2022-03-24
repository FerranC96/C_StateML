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
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_text
from joblib import dump
from aux import *
import seaborn as sns; sns.set()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~I/O~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
folder_name = "trainRF"

if os.path.isdir(f"./output/{folder_name}") == False:
    os.makedirs(f"./output/{folder_name}")
#INput dir to get train and validation data (downsampled). Second_dir for test2
train_dir = "../D_CyTOF/C_Fig4Time/States" #using timepoint data from SI orgs NatMethods Fig4
test2_dir = "../D_CyTOF/CRC-TME/Epithelial-Cells" #Test against colon organoids from natmethods Fig5
# second_dir = input_dir
output_dir = f"./output/{folder_name}"

info_run =  input("Write RF info run (using no spaces!): ")
if os.path.isdir(f"{output_dir}/{info_run}") == True:
    print("THIS INFO RUN HAS ALREADY BEEN USED. Be aware of overwritting data")
    # sys.exit("CLOSING! Uncomment line if you want to procede")
else:
    os.makedirs(f"{output_dir}/{info_run}")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


#~~~~~~~~~~~~~~~~~~~~~Prepare TRAIN and validation data~~~~~~~~~~~~~~~~~~~~#

#TURN INTO A FUCNITON INCORPORATING CELL SATES LABEL FISHING FFROM FILENAMES
dTrain = pd.DataFrame()

filelist = [f for f in os.listdir(train_dir) if f.endswith(".txt")]
print ("Input files:")
if len(filelist) ==1:
    print("Only one")
    print (filelist[0])
    df = pd.read_csv(f"{train_dir}/{filelist[0]}", sep = "\t")#change to "," asap
    if "cell-state_num" not in df.columns:
        sys.exit("ERROR: NO CELL STATE INFO (missing column called 'cell-state_num')")
    dTrain = pd.concat([dTrain, df], ignore_index=True)
elif len(filelist)>1:
    print(f"{len(filelist)} files:")
    fcounter = 0
    for i in filelist: #TODO: Update with newer snippets from CyGNAL
        print (i)
        name = i.split('.txt')[0]
        fcounter += 1
        df = pd.read_csv(f"{train_dir}/{i}", sep = "\t")
        df["file_origin_trainRFcs"] = str(fcounter)+"_|_"+ name
        # df["Sample_ID-Cell_Index"] = df["Cell_Index"].apply(
        #                                 lambda x: str(fcounter)+"-"+str(x))
        if "cell-state_num" not in df.columns:
            print("Train data missing cell state information. Trying to label if info present in file name")
            if "apoptosis" in i.lower():
                df["cell-state"] = "apoptosis"
                df["cell-state_num"] = "0"
            elif "g0" in i.lower():
                df["cell-state"] = "g0"
                df["cell-state_num"] = "1"
            elif "g1" in i.lower():
                df["cell-state"] = "g1"
                df["cell-state_num"] = "2"
            elif "s-phase" or "s_phase" in i.lower():
                df["cell-state"] = "s-phase"
                df["cell-state_num"] = "3"
            elif "g2" in i.lower():
                df["cell-state"] = "g2"
                df["cell-state_num"] = "4"
            elif "m-phase" or "m_phase" in i.lower():
                df["cell-state"] = "m-phase"
                df["cell-state_num"] = "5"
            else:
                print(f"ERROR: File {i} could not be assigned to a cell state. Check your file names!")
        dTrain = pd.concat([dTrain, df], ignore_index=True)
else:
    sys.exit(f"ERROR: There are no .txt files in {train_dir}!")


#Abs to use
cols = read_marker_csv(train_dir)
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


#Sanity check for presence of cell-state columns:
if "cell-state" not in dTrain.columns:
    sys.exit("ERROR: NO CELL STATE INFO (missing column called 'cell-state_num')")
    

#Downsampling section
if df.equals(dTrain["cell-state_num"].value_counts()):
    print("JACK-POT! Downsampling to balance classes not required")
    dTrain_dwnsBalanced = dTrain
else:
    print ("Downsampling taking place. Check output folder for more info")
    print (dTrain["cell-state_num"].value_counts())
    dTrain_dwnsBalanced = downsample_data(dTrain, f"dwnsBalanced_b4_RF", 
                f"{output_dir}/{info_run}", 
                split_bycol="cell-state_num")

print (dTrain_dwnsBalanced["cell-state_num"].value_counts())

#Translation section: Not strcilty needed as panel markers should have been 
# generated with this datasets

dTrain_dwnsBalanced.columns = [i.replace(" ","_") for i in dTrain_dwnsBalanced.columns]

#TRANSLATION -> needs to become a function ASAP
translation = {} #Translation layer needed since it's uncommon for names to be exact matches.

for i in dTrain_dwnsBalanced.columns:
    if i in cols:
        print("Exact match")
        print (i)
        translation[i] = i
    else:
        for i2 in cols:
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
                        print("Marker with shorter name in model")
                        print("from model", i2)
                        print("from input data", i)
                        translation[i] = i2
            except:
                pass
print(translation)

if len(translation) != len(cols):
    sys.exit(f"ERROR: Missing model features in {filelist[0]}!")
    #IN the future intead of sys.exit warn user and try to use the minimal 5 marker model.


dTrain_dwnsBalanced = dTrain_dwnsBalanced.rename(columns=translation)
dTrain_dwnsBalanced = dTrain_dwnsBalanced.drop(columns=np.setdiff1d(dTrain_dwnsBalanced.columns, cols))
dTrain_dwnsBalanced


#Section to check if data has been arcsinh trasnformed:
if dTrain_dwnsBalanced.max().max() > 12:
    print("WARNING! IS YOUR DATA NORMALISED? \n CyTOF data is generally normalised using arcsinh(a=5), and it seems like your data might not have been normalised.")
    #Eventually have some code to prompt the user to normalise it here if wanted



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

mRF_cvscores = cross_val_score(mRF, X, Y)
print(f"CV score of base model: {mRF_cvscores.mean()} (+/- {mRF_cvscores.std()*2})")

print(metrics.classification_report(mRF_preds,Y_test))
print(metrics.classification_report(mRFcal_preds,Y_test))



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
plt.savefig(f"{output_dir}/{info_run}/{info_run}_feature_importances.png", bbox_inches = "tight")

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
print("DEPRECATED SCRIPT. Used to generate the RF cycle classifier models") 

#Get non-downs data
# processed_alldf = concat[cols].copy()


#~~~~~~~~~~~~~~~~~~~~~Prepare TEST data~~~~~~~~~~~~~~~~~~~~#
#UPDATE BELOW WITH NEW FORMAT AND FUNCTIONS
print(test2_dir)

filelist2 = [f for f in os.listdir(second_dir) if f.endswith(".txt")]
if len(filelist2) == 0:
    sys.exit(f"ERROR: There are no .txt files in {second_dir}!")
#Check the files found in the directory:
print ("Input files in second_dir:")
for i in filelist2:
    print (i)

test_df = label_concatenate(filelist2, second_dir)

test_df = test_df[cols] #Must use shared collumns bwten train and test data





#~~~~~~~~~~~~~~~~~~~~~TEST Random Forest~~~~~~~~~~~~~~~~~~~~#

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

print("Cross validation scores for test data (mean and 95% CI):")
cv_scores_test = cross_val_score(model_RFreg, X_test, y_test)
print(f"Accuracy: {cv_scores_test.mean()} (+/- {cv_scores_test.std()*2})")


#~~~~~~~~~~~~~~~~~~~~~TEST/validation metrics and plots~~~~~~~~~~~~~~~~~~~~#

# print (metrics.r2_score(y_test, predictions))
print ("Score agains validation data", model_RFreg.score(X_valid, y_valid))
print(metrics.classification_report(predictions,y_valid))
json.dump(metrics.classification_report(predictions,y_valid, output_dict=True), 
            open(f"{output_dir}/TRAINING_{info_run}/{info_run}_ClassReport_VALIDdata_{model_RFreg.score(X_valid, y_valid)}.json", "w"))
print ("Predictions on test data: ", model_RFreg.score(X_test, y_test))
print(metrics.classification_report(predict_test,y_test))
json.dump(metrics.classification_report(predict_test,y_test, output_dict=True), 
            open(f"{output_dir}/TRAINING_{info_run}/{info_run}_ClassReport_TESTdata_{model_RFreg.score(X_test, y_test)}.json", "w"))

#Log loss:
logloss_score = metrics.log_loss(y_test, clf.predict_proba(X_test))
print("Log loss score is:", logloss_score)

###########
print(metrics.classification_report(sig_clf.predict(X_test),y_test))
logloss_score = metrics.log_loss(y_test, sig_clf.predict_proba(X_test))
print("Log loss score for sigmoid is:", logloss_score)
############

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


