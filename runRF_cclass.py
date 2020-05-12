import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from joblib import load
from sklearn import metrics
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree.export import export_text
from aux import *
import seaborn as sns; sns.set()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~CONFIG~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
folder_name = "classRF"

if os.path.isdir(f"./output/{folder_name}") == False:
    os.makedirs(f"./output/{folder_name}")

    
input_dir = "../D_CommonDatasets/CRC-TME/ALLcells"
output_dir = f"./output/{folder_name}"

info_run =  input("Write RF info run (using no spaces!): ")
os.makedirs(f"{output_dir}/{info_run}")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Choose RF MODEL~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#Default models
    #Dsiplay our 3 models (epi, macro, fibro) +MIX?
    #Ask user which type of cells: Epithelial, macro, fibro, MIX
    #Grab via lowercare(epi/macro/fibro or mix) in user input
        #Load correspoding filename
#Custom model:
    #Eventually let users provide their own labelled datasets
        #Clearly specify format
    #Read, generate model from that
        #Check test_RF for the code-generating model
        #Print the performacne of model and confusion matrix et al

#Currently hard reading from model folder:
model_to_use = "./Models/EPI_msiFig4_RFcclass.joblib"
print("Using model: ", model_to_use)
clf = load(model_to_use)



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~READ INPUT DATA~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#Specific format. Shouwl contain the following columns:
    #CC markers, as of now with the Isotopes we used to gen model, 
    #so I would need to filter isotopes first or use custom names

#Load and read input data: Concat all inputs 
input_data = pd.DataFrame()
#Add counter to keep track of the number of files in input -> 
# -> cell ID will be a mix of these (Filenumber | filename.txt)
fcounter = 0
filelist = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
print ("Input files:")
for i in filelist: #TODO: Update with newer snippets from CytofDatanAnalysis
    print (i)
    name = i.split('.txt')[0]
    fcounter += 1
    df = pd.read_csv(f"{input_dir}/{i}", sep = '\t')
    df["file_origin"] = str(fcounter)+" | "+ i # 
    df["Sample_ID-Cell_Index"] = df["Cell_Index"].apply(
                                    lambda x: str(fcounter)+"-"+str(x))
    input_data = input_data.append(df, ignore_index=True)



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~IMPORTANT NOTICE~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#CURRENT IMPLEMENTATION: Input data is already labelled and thus we can 
#generate prediction reports

#Load markers to use: -> In the future these should really be hardcodedin
cols = read_marker_csv(input_dir)
cols.append("cell-state_num") 

#Filter input_data
working_data = input_data[cols]



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~State PREDICTION~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#PREDICT STATES ON INPUT DATA

#Apply class and get predicted cell states:
X_all = working_data.drop("cell-state_num", axis=1) #No need for this once we have hardcoded cycle markers
predict_alldata = clf.predict(X_all)

#Save results to file:
input_data["Predicted_state"] = predict_alldata
# print ("Save predictions as a tab-separated .txt file")
# input_data.to_csv(f"{output_dir}/{info_run}/{info_run}_predictions.txt", 
#                     index = False)



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~SCORING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#Only possible if data had "true labels" -> SEE IMPORTANT NOTICE
#PREDICTON SCORES:Compare predictions against labels
y_all = working_data["cell-state_num"]
class_report = metrics.classification_report(predict_alldata,y_all)
print(class_report)
print(type(class_report))

f = open(f"{output_dir}/{info_run}/{info_run}_classification_report.txt", 'w')
f.write(class_report)
f.close()




#Plot tree number 5 out of the whole forest
estimator = clf.estimators_[5]
from sklearn.tree import export_graphviz
# Export as dot file
export_graphviz(estimator, out_file=f"{output_dir}/{info_run}/{info_run}_tree.dot",
                feature_names=X_all.columns,
                class_names=["apoptosis","G0","G1","S","G2","M"],
                rounded = True, proportion = False, 
                precision = 2, filled = True)


#Store confusion matrix data
mat = metrics.confusion_matrix(y_all, predict_alldata)
pd.DataFrame(mat).to_csv(f"{output_dir}/{info_run}/{info_run}_confusion_matrix.csv")

#Plot confussion matrix
plt.figure()
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.savefig(f"{output_dir}/{info_run}/{info_run}_confusion_matrix.png")
plt.show()

