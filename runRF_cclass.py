import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import tasklogger
from joblib import load
from sklearn import metrics
import seaborn as sns; sns.set()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~CONFIG~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
folder_name = "classRF"

if os.path.isdir(f"./output/{folder_name}") == False:
    os.makedirs(f"./output/{folder_name}")

    
input_dir = "./input"
output_dir = f"./output/{folder_name}"

info_run =  input("Write RF info run (using no spaces!): ")
os.makedirs(f"{output_dir}/{info_run}")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Choose RF MODEL~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

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

#Currently hard reading from model folder:
model_to_use = "./Models/EPI_8_stateMarkers/PDO21_UNTrep1_sP_tUNTrep2_RFcclass.joblib" 
print("Using model: ", model_to_use)
clf = load(model_to_use)

#Model used -> Nice to have as a sanity check
#Note that the models generated in 2020 used sklearn 0.2x, whereas we are now in sklearn 1.0.x...
#This could cause issues so the more appropiate approach would be to redo them 
# (or at least the PDO 8 marker one) to ensure a newer version is used.

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~READ INPUT DATA~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#Specific format. Should contain the following columns:
    #CC markers, as of now with the Isotopes we used to gen model, 
    #so I would need to filter isotopes first or use custom names

#Load and read input data: Concat all inputs 
input_data = pd.DataFrame()
#Add counter to keep track of the number of files in input -> 
# -> cell ID will be a mix of these (Filenumber | filename.txt)
fcounter = 0
filelist = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
print ("Input files:")
if len(filelist) ==1:
    print("Only one")
    print (filelist[0])
    df = pd.read_csv(f"{input_dir}/{filelist[0]}", sep = ',')
    input_data = input_data.append(df, ignore_index=True)

elif len(filelist)>1:
    for i in filelist: #TODO: Update with newer snippets from CyGNAL
        print (i)
        name = i.split('.csv')[0]
        fcounter += 1
        df = pd.read_csv(f"{input_dir}/{i}", sep = ',')
        df["file_origin"] = str(fcounter)+" | "+ i # 
        df["Sample_ID-Cell_Index"] = df["Cell_Index"].apply(
                                        lambda x: str(fcounter)+"-"+str(x))
        input_data = input_data.append(df, ignore_index=True)
else:
    print("ERROR NO FILES")


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~IMPORTANT NOTICE~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# #CURRENT IMPLEMENTATION: Input data is already labelled and thus we can 
# #generate prediction reports

# #Load markers to use: -> In the future these should really be hardcoded in
# cols = read_marker_csv(input_dir)
# cols.append("cell-state_num") 

#Filter input_data

#GET FEATURE NAMES IN MODEL:
#Current implementaion is manual, since the info isin't stored in the model.
# This is however crucial, since we need to know the features AND THEIR ORDER 
# used in the train data of the model

#MANUAL APPROACH
#The model used (./Models/EPI_5_stateMarkers/EPI_msiFig4_RFcclass.joblib) uses just 5 markers (and not the old approach of 5 states+2 top ptms)
#THe markers used have spaces instead of _ after in ccaspase and cyclin and I made the space->_ change in cols
cols = ["150Nd_pRB_S807_S811_v2", "127I_IdU", "176Yb_Cyclin_B1_2_v2", 
        "89Y_pHH3_S28", "163Dy_cPARP_D214_2", "152Sm_pAKT_T308_v6",
        "158Gd_pP38_T180_Y182", "142Nd_cCaspase_3_D175_v3", "143Nd_Geminin",
        "169Tm_PLK1"] 
#AUTOMATED APPROACH: Relies on feature names being saved during training and 
# dumped togehter with the model. Then we can load them in here.
#In theory starting with sklearn 1.0 the RFmodel should have .feature_names_in_ 
# attribute that states features seen during fitting!


#TRANSLATION
translation = {} #Translation layer needed since it's uncommon for names to be exact matches.

for i in input_data.columns:
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

#WENRperm with murine 5 state model:
# translation = {'89Y_pHH3': '89Y_pHH3', 
#                 '127I_IdU': '127I_IdU', 
#                 '142Nd_cCaspase_3': '142Nd_cCaspase_3', 
#                 '150Nd_pRB': '150Nd_pRB', 
#                 '173Yb_Cyclin_B1': '176Yb_Cyclin_B1'}
# translation = {'89Y_pHH3': '89Y_pHH3', #Change vals to undo space2_ conversion
#                 '127I_IdU': '127I_IdU', 
#                 '142Nd_cCaspase_3': '142Nd_cCaspase 3', 
#                 '150Nd_pRB': '150Nd_pRB', 
#                 '173Yb_Cyclin_B1': '176Yb_Cyclin B1'}
# cols = ["150Nd_pRB", "142Nd_cCaspase 3", "127I_IdU", "176Yb_Cyclin B1",
#         "89Y_pHH3"] #Undo changes to cols to 100% match model features

# #WENRperm with PDO21 8 state model: -> CAN'T RUN SINCE MODEL Abs are missing from WENRperm panel
# translation = {'89Y_pHH3': '89Y_pHH3_S28', '127I_IdU': '127I_IdU', 
#                 '142Nd_cCaspase_3': '142Nd_cCaspase_3_D175_v3', 
#                 '150Nd_pRB': '150Nd_pRB_S807_S811_v2', 
#                 '152Sm_pAKT_T308': '152Sm_pAKT_T308_v6', 
#                 '173Yb_Cyclin_B1': '176Yb_Cyclin_B1_2_v2'}


#FORCE MODEL FEATURE ORDER IN INPUT DATA




working_data = input_data.rename(columns=translation)




working_data = working_data.drop(columns=np.setdiff1d(working_data.columns, cols))
working_data



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~State PREDICTION~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#PREDICT STATES ON INPUT DATA

#Apply class and get predicted cell states:
X_all = working_data

#Denoise input testing data with MAGIC
# print(X_all.head())
# with tasklogger.log_task("allcore_magic"):
#     magic_op = magic.MAGIC(knn=5, solver="exact" , n_jobs=-1)
#     X_alldenoised = magic_op.fit_transform(X_all)
# print(X_alldenoised)
# X_all = X_alldenoised

predict_alldata = clf.predict(X_all)

#Save results to file:
input_data["Predicted_state"] = predict_alldata
print ("Save predictions as a tab-separated .txt file")
input_data.to_csv(f"{output_dir}/{info_run}/{info_run}_predictions.csv", 
                    index = False)

#Plot tree number 5 out of the whole forest
estimator = clf.estimators_[5]
from sklearn.tree import export_graphviz
# Export as dot file
export_graphviz(estimator, out_file=f"{output_dir}/{info_run}/{info_run}_tree.dot",
                feature_names=X_all.columns,
                class_names=["apoptosis","G0","G1","S","G2","M"],
                rounded = True, proportion = False, 
                precision = 2, filled = True)

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

