import sys
import glob
import numpy as np
import os
#from rdkit import Chem
import pandas as pd
#from rdkit.Chem.PandasTools import LoadSDF
import matplotlib.pyplot as plt
import random
import hdbscan
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#from rdkit.Chem import Descriptors, MACCSkeys
#from rdkit.Chem import AllChem
from hdbscan import HDBSCAN
from expt2_feature_filtering import *
#Danny Ruiz de Castilla 02.28.2025

#pcaDict = {canonicalSmiles : [ yield]} for local Chemistries 
#smilesDict = {canonicalSmiles : [ xPCA , yPCA]}
def dimensionalityReduction(X , smiles):

    X.to_csv("test.csv")
    scaler = StandardScaler()
    scaledX = scaler.fit_transform(X)


    pca = PCA(n_components = 2 ,svd_solver="full" )
    xPCA = pca.fit_transform(scaledX)

    # Get PCA loadings
    loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=X.columns)

    # Sort features by absolute contribution to PC1 and PC2
    top_features_pc1 = loadings['PC1'].abs().sort_values(ascending=False)
    top_features_pc2 = loadings['PC2'].abs().sort_values(ascending=False)

    top_features = pd.DataFrame({
        "Feature": top_features_pc1.index,
        "PC1_Contribution": top_features_pc1.values,
        "PC2_Contribution": top_features_pc2.loc[top_features_pc1.index].values  # Align PC2 with PC1 sorting
    })

    # Save to a .dat file if it does not exist
    file_path = "top_features.csv"
    if not os.path.exists(file_path):
        top_features.to_csv(file_path, sep="\t", index=False)

    dfPCA = pd.DataFrame(xPCA, columns = ["PCA1" , "PCA2"] )
    xAxis = list(dfPCA["PCA1"])
    yAxis = list(dfPCA["PCA2"])
    pcaDict = {}
    for i in range (len(smiles)):
        smile = smiles[i]
        pcaDict[smile] = [xAxis[i] , yAxis[i]]
    return pcaDict   
def findHighlights(smilesDict, partition1, chemistryDict):
    highlightDict = {}
    to_remove = []

    for substrate, parameter in chemistryDict.items():
        print(substrate , parameter)
        if substrate in smilesDict and parameter >= partition1:
            #print("MATCH")
            xPCA, yPCA = smilesDict[substrate]  # Direct unpacking
            highlightDict[substrate] = [xPCA, yPCA]
            to_remove.append(substrate)

    # Remove keys after iteration to avoid modifying the dictionary while looping
    for key in to_remove:
        del smilesDict[key]

    return highlightDict, smilesDict


def makePlots(pcaDict ,  partitionList , chemistryDicts  ,chemistryStr , colors , partitionStr):
    
    for partition in partitionList:
        print("partition" , partition)
        for i , chemistryDict in enumerate(chemistryDicts): #this is the highlighted chemistry in the plot
            highlightDict , smilesDict = findHighlights(pcaDict.copy()  ,  partition , chemistryDict )
            print(highlightDict)
            try:
                xHigh , yHigh = zip (*highlightDict.values())
                colorScatter = True
            except ValueError:
                colorScatter = False

            xBland , yBland = zip(*smilesDict.values())
            chemistryLabel = chemistryStr[i]
            print("**********************" + str(chemistryLabel))

            xLabel = "PCA1"
            yLabel = "PCA2"
            dpi = 300
            plt.figure(figsize=(800 / dpi, 600 / dpi), dpi=dpi)  # 800x600 pixels
            if colorScatter:
                plt.scatter(list(xHigh), list(yHigh), c=colors[i], alpha=0.7 , s=10)
            plt.scatter(list(xBland), list(yBland), c="grey", alpha=0.14 , s=10)
            plt.xlabel(xLabel, fontsize=16, fontweight='bold', color='black')
            plt.ylabel(yLabel, fontsize=16, fontweight='bold', color='black')
            plt.title("Substrate Scope for " + str(chemistryLabel) + " at " + str(partition) + "%" + str(partitionStr), fontsize=18, fontweight='bold', color='navy')
            plt.xticks(fontsize=14, color='black')
            plt.yticks(fontsize=14, color='black')
            #plt.grid(True, linestyle='--', alpha=0.5)
            plt.savefig(str(chemistryLabel) + "at" + str(partition) + ".png", dpi=300, bbox_inches='tight')
    
def featureFiltering(outDir , X , feature_labels , featureStr):
    if not os.path.exists(outDir + "/" + str(featureStr) + "featureFiltering.dat"):
        with open(outDir + "/" + str(featureStr) + "featureFiltering.dat", "w") as f:
            f.write(f"Total starting feature count: {len(feature_labels)}")
            f.write("".join([f'\n\t{label}' for label in feature_labels]))
            
            X, feature_labels, dropped_features = remove_by_variance(X, feature_labels)
            print("90" , type(X))

            text = "\n\n\nFeatures drop due to low variance: " + "".join([f'\n\t{label}' for label in dropped_features])
            f.write(text)

            X, feature_labels, drop_group = correlation_analysis(X, feature_labels, threshold=0.95)
            text = "\n\n\nFeatures drop due correlation: (STILL HAS ISSUES) "
            print("97" , type(X))
            import json
            text += json.dumps(drop_group, indent=4).replace('\n', '\n\t')
            f.write(text)

            X, feature_labels  = spearmanr_correlation(X, feature_labels, threshold=0.95)
            print("103" , type(X))
    return X , feature_labels
def locateNans(df):
    columns = df.columns.tolist()
    nanDict = {}
    for column in columns:
        col = df[column]
        indCol = col.isna().to_numpy().nonzero()[0]
        if len(indCol) != 0:
            #print("NANS" , indCol)
            nanDict[str(column)] = list(indCol)

    return nanDict
def eliminateNans(df, nanDict):
    allNanRows = set()
    for key in nanDict.keys():
        rows = nanDict[key]
        allNanRows.update(rows)
    
    df = df.drop(list(allNanRows))
    df = df.reset_index(drop=True)
    return df
def transformations(dataframeDirs):
    usualSuspects = ["SMILES" , "Compound_Name", "%Yields", "ChemistryType"  , "Unnamed: 0"]

    dataframeMast = pd.DataFrame()
    
    # Check if input is valid
    if not dataframeDirs:
        return pd.DataFrame(), pd.Series()
    
    try:
        # Process each file
        for dfDir in dataframeDirs:
            try:
                df = pd.read_csv(dfDir)
                
                # Handle empty files
                if df.empty:
                    print(f"Warning: Empty file {dfDir}")
                    continue
                    
                if dataframeMast.empty:
                    dataframeMast = df.copy()
                else:
                    commonCols = list(set(df.columns) & set(dataframeMast.columns))
                    if not commonCols:
                        print(f"Warning: No common columns with file {dfDir}")
                        continue
                    dataframeMast = pd.concat([dataframeMast[commonCols], df[commonCols]], ignore_index=True)
            
            except Exception as e:
                print(f"Error processing file {dfDir}: {e}")
        
        # Check if master dataframe is populated
        if dataframeMast.empty:
            return pd.DataFrame(), pd.Series()
        
        # Save SMILES column if it exists
        if 'SMILES' in dataframeMast.columns:
            smileList = dataframeMast['SMILES'].copy()

        else:
            print("Warning: 'SMILES' column not found in the dataframe")
            smileList = pd.Series()
            
        for str in usualSuspects:
            if str in list(dataframeMast.columns):
                dataframeMast = dataframeMast.drop(str, axis=1)

        return dataframeMast, smileList
        
    except Exception as e:
        print(f"Critical error in transformations: {e}")
        return pd.DataFrame(), pd.Series()
def createCSV(dataFrame , saveDir ,saveStr):
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    dataFrame.to_csv(saveDir + "/" + str(saveStr) + ".csv", index=True) 
    return dataFrame
if __name__ == "__main__":

    chemistryDirs = str(sys.argv[1])
    datasetDir = str(sys.argv[2])
    chemType = str(sys.argv[3])
    #make sure to adjust what Chemistries you are plotting
    partitionList = [50 , 70 , 85 , 90]
    colorList = ['red' , 'blue' , 'green' , 'yellow']
    initdataSets = glob.glob(datasetDir + "/*.csv")
    substrateSpaces = glob.glob(chemistryDirs + "/*.csv")
    initdataSets = sorted(initdataSets)
    Xdataframe , smileList = transformations(initdataSets )
    nanDict = locateNans(Xdataframe)
    if len(nanDict) != 0:
        Xdataframe["SMILES"] = smileList
        Xdataframe = eliminateNans(Xdataframe , nanDict)
    nanDict = locateNans(Xdataframe)

    smileList = Xdataframe["SMILES"].copy()
    Xdataframe = Xdataframe.drop("SMILES", axis=1)


    featureLabels = list(Xdataframe.columns)
    
    X , featureLabels  = featureFiltering(datasetDir , Xdataframe ,featureLabels , chemType)

    nanDict = locateNans(X)
    print("nansWayAfter" , len(nanDict))
    print("180" , type(X))

    pcaDict = dimensionalityReduction(X , smileList)

    chemSpaceDict = []
    chemDirList = []
    for dir in substrateSpaces:
        #print(dir)
        df = pd.read_csv(dir)
        smileList = list(df["SMILES"])
        yieldList = list(df["Yield"])
        chemistryDict = dict(zip(smileList, yieldList))
        chemSpaceDict.append(chemistryDict)
        split1 = dir.split("/")[-1]

        split2 = split1.split(chemType)[0]
        chemDirList.append(split2)
    makePlots(pcaDict.copy() ,  partitionList , chemSpaceDict  , chemDirList, colorList , "Yield")
    

    

    





    





