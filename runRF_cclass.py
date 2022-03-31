import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from joblib import load
from sklearn import metrics
import seaborn as sns
from sklearn.semi_supervised import LabelSpreading; sns.set()
from aux import yes_or_NO, readRFcs_data, translateAbMarkers, arcsinh_transf

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~CONFIG~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
folder_name = "classRF"

if os.path.isdir(f"./input/{folder_name}") == False:
    os.makedirs(f"./input/{folder_name}")
if os.path.isdir(f"./output/{folder_name}") == False:
    os.makedirs(f"./output/{folder_name}")

    
input_dir = f"./input/{folder_name}"
output_dir = f"./output/{folder_name}"

#Default models
    #Dsiplay our 3 models (epi, macro, fibro) +MIX?
    #Ask user which type of cells: Epithelial, macro, fibro, MIX
    #Grab via lowercase(epi/macro/fibro or mix) in user input
        #Load correspoding filename
#Custom model:
    #Eventually let users provide their own labelled datasets
        #Clearly specify format
    #Read, generate model from that
        #Check test_RF for the code-generating model
        #Print the performacne of model and confusion matrix et al

#Currently hard reading trainRf output, eventually select model from model folder:
model_to_use = "EPI_5s_mcNatM5_RFcclass" 
print("Using model: ", model_to_use)
clf = load(f"./output/trainRF/EPI_5s_mcNatM5/{model_to_use}.joblib")

#Model used -> Nice to have as a sanity check
#Note that the models generated in 2020 used sklearn 0.2x, whereas we are now in sklearn 1.0.x...
#This could cause issues so the more appropiate approach would be to redo them 
# (or at least the PDO 8 marker one) to ensure a newer version is used.

print(clf)
print(clf.n_features_in_, clf.feature_names_in_)
print(clf.classes_, "Apoptosis, G0, G1, S-phase, G2, M-phase")


info_run =  input("Write info run (using no spaces!): ")
if len(info_run) == 0:
    print("No info run given. Saving results in UNNAMED")
    info_run = "UNNAMED"
else:
    info_run = f"{info_run}_[{model_to_use}]"
    print("Info run (with model)", info_run)

if os.path.isdir(f"{output_dir}/{info_run}") == False:
    os.makedirs(f"{output_dir}/{info_run}")
else:
    if info_run !="UNNAMED":
        sys.exit("ERROR: You already used this name for a previous run. \nUse a different name!")


print("This script is used to label cell states using a pre-built model.\n",
    "It can either label an unlabelled dataset or generate predictions on an alreay labelled dataset and compare the results")
labels_input = yes_or_NO("Is your input data already labelled with cell states?")
if labels_input:
    print("\nMultiple labelled datasets are accepted as long as the cell state information is stored in a column called 'cell-state_num'")
    print("Otherwise, if the labels column is missing, we can still generate the labels provided that each file contains cells in a single state and that is reflected in the name.",
    "Should that be the case all files in the input folder will be joined together.")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~READ INPUT DATA~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#Specific format. Should contain the following columns:
    #CC markers, as of now with the Isotopes we used to gen model, 
    #so I would need to filter isotopes first or use custom names

#In this section we load the single dataset found in the input folder

labels_input, dInput_list = readRFcs_data(labels_input, 
                                            input_dir, 
                                            ext=".txt", sep="\t")

print(labels_input)
[print(i) for i in dInput_list]


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~IMPORTANT NOTICE~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#GET FEATURE NAMES IN MODEL:
#need to know the features AND THEIR ORDER 
# used in the train data of the model

features_model = list(clf.feature_names_in_)
print("The model has the following features: ", features_model)

#Translation:
dTrans_list = []
[print(i.columns) for i in dInput_list]
for i in range(len(dInput_list)):
    dInput_list[i].columns = [col.replace(" ","_") for col in dInput_list[i].columns] #Columns should NOT have spaces!
    dTrans_list.append(translateAbMarkers(dInput_list[i], features_model))
[print(i.columns) for i in dTrans_list]

#Filter the dTrans to avoid issues with transformation et al:
if labels_input==True:
    for i in range(len(dTrans_list)):
        dTrans_list[i] = dTrans_list[i][features_model+["cell-state_num"]]
else:
    for i in range(len(dTrans_list)):
        dTrans_list[i] = dTrans_list[i][features_model]

#Section to check if data has been arcsinh transformed
for i in range(len(dTrans_list)):
    if dTrans_list[i].select_dtypes(include="number").max().astype("float64").max() > 12:
        #No need to use select_dtypes(include="number") ?
        print("WARNING! IS YOUR DATA NORMALISED? \n CyTOF data is generally normalised using arcsinh(a=5), and it seems like your data might not have been normalised.")
        norm_data = yes_or_NO("Shall we normalise the data now?", 
                                default="YES")
        if norm_data==False:
            stop_script = yes_or_NO("Do you want to exit the script then?")
            if stop_script:
                sys.exit("Exiting the script")
        else: #Normalise, shouldn't affect cell-state_num col ->[0] is the Df
            dTrans_list[i] = arcsinh_transf(cofactor=5, no_arc=dTrans_list[i])[0]

[print(i) for i in dTrans_list]


#PREDICT STATES ON INPUT DATA

#Apply class and get predicted cell states:

#MAGIC
# print(X_all.head())
# with tasklogger.log_task("allcore_magic"):
#     magic_op = magic.MAGIC(knn=5, solver="exact" , n_jobs=-1)
#     X_alldenoised = magic_op.fit_transform(X_all)
# print(X_alldenoised)
# X_all = X_alldenoised

#Predict no matter labels_input status:
dPred_list = []
for i in range(len(dTrans_list)):
    if labels_input:
        dPred_list.append(clf.predict(dTrans_list[i].drop(columns="cell-state_num")))
    else:
        dPred_list.append(clf.predict(dTrans_list[i]))

[print(i) for i in dPred_list]


#Add predictec label to riginal dataframe and save results to file
print ("Save predictions as a comma separated .csv file")
for i in range(len(dPred_list)):
    dInput_list[i]["cell-state_prediction"] = dPred_list[i]
    dInput_list[i].to_csv(#If dealing with multiple df, it will try to name based on number of cells. Potentially an issue when input is mutliple unrelated downsampled dataframes
        f"{output_dir}/{info_run}/d{dInput_list[i].shape[0]}_predictions.csv",
        index = True)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~SCORING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#Only possible if data had "true labels" -> SEE IMPORTANT NOTICE
#PREDICTON SCORES:Compare predictions against labels

if labels_input:
    for i in range(len(dPred_list)):
        print (metrics.accuracy_score(dTrans_list[i]["cell-state_num"],
                                            dPred_list[i]))
        print (metrics.balanced_accuracy_score(dTrans_list[i]["cell-state_num"],
                                            dPred_list[i]))
        print (metrics.classification_report(dTrans_list[i]["cell-state_num"],
                                            dPred_list[i]))
        print (metrics.confusion_matrix(dTrans_list[i]["cell-state_num"],
                                            dPred_list[i]))
        print (metrics.f1_score(dTrans_list[i]["cell-state_num"],
                                            dPred_list[i],
                                average="micro"))
        print (metrics.multilabel_confusion_matrix(dTrans_list[i]["cell-state_num"],
                                            dPred_list[i]))


# y_all = working_data["cell-state_num"]
# class_report = metrics.classification_report(predict_alldata,y_all)
# print(class_report)
# print(type(class_report))

# f = open(f"{output_dir}/{info_run}/{info_run}_classification_report.txt", 'w')
# f.write(class_report)
# f.close()

# #Store confusion matrix data
# mat = metrics.confusion_matrix(y_all, predict_alldata)
# pd.DataFrame(mat).to_csv(f"{output_dir}/{info_run}/{info_run}_confusion_matrix.csv")

# #Plot confussion matrix
# plt.figure()
# sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False)
# plt.xlabel('true label')
# plt.ylabel('predicted label')
# plt.savefig(f"{output_dir}/{info_run}/{info_run}_confusion_matrix.png")
# plt.show()

