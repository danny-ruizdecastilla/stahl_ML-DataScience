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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from featureMaping import savePNG , createCSV , locateNans
from expt2_feature_filtering import *
#Danny Ruiz de Castilla 02.28.2025

#pcaDict = {canonicalSmiles : [ yield]} for local Chemistries 
#smilesDict = {canonicalSmiles : [ xPCA , yPCA]}
def dimensionalityReduction(X , smiles):
    np.random.seed(42)
    X.to_csv("test.csv")
    scaler = StandardScaler()
    scaledX = scaler.fit_transform(X)


    pca = PCA(n_components = X.shape[1] ,svd_solver="full" )
    pca.fit_transform(scaledX)


    explainedVar = pca.explained_variance_ratio_

    #print("explained variance ratio:", explainedVar, f"(total):{np.sum(pca.explained_variance_ratio_)}")
    top2 = np.argsort(explainedVar)[-2:][::-1] 
    #print(top2)
    top2PCA = pca.components_[top2]
    xPCA_ = scaledX @ top2PCA.T
    #print(xPCA_)
    loadings = pd.DataFrame(pca.components_.T * np.sqrt(pca.explained_variance_), columns=[f'PC{i+1}' for i in range(pca.n_components_)] , index = X.columns)

    top_features_pc1 = loadings['PC1'].abs().sort_values(ascending=False)
    top_features_pc2 = loadings['PC2'].abs().sort_values(ascending=False)

    top_features = pd.DataFrame({
        "Feature": top_features_pc1.index,
        "PC1_Contribution": top_features_pc1.values,
        "PC2_Contribution": top_features_pc2.loc[top_features_pc1.index].values  # Align PC2 with PC1 sorting
    })

    file_path = "topFeatures.csv"
    if not os.path.exists(file_path):
        top_features.to_csv(file_path, sep="\t", index=False)

    dfPCA = pd.DataFrame(xPCA_, columns = ["PCA1" , "PCA2"] )
    #print(dfPCA)


    xAxis = list(dfPCA["PCA1"])
    yAxis = list(dfPCA["PCA2"])
    dfPCA["SMILES"] = smiles
    kmeansMAST = KMeans( n_clusters = specialK , random_state = 42)
    kmeansMAST.fit(xPCA_)
    dfPCA['Clusters'] = kmeansMAST.labels_
    createCSV(dfPCA, datasetDir , chemType+ "_clustered")
    savePNG(dfPCA , datasetDir , chemType + "_clustered" , "PCA" , cluster = True)

    pcaDict = {}
    for i in range (len(smiles)):
        smile = smiles[i]
        pcaDict[smile] = [xAxis[i] , yAxis[i]]
    return pcaDict   , list(kmeansMAST.labels_)
def findHighlights(smilesDict, partition1, chemistryDict):
    highlightDict = {}
    hollowDict = {}
    to_remove = []

    for substrate, parameter in chemistryDict.items():
        #print(substrate , parameter)
        if substrate in smilesDict and parameter >= partition1:
            #print("MATCH")
            xPCA, yPCA = smilesDict[substrate]  # Direct unpacking
            highlightDict[substrate] = [xPCA, yPCA]
            to_remove.append(substrate)

        elif substrate in smilesDict and parameter < partition1: #hollow points
            hollowX , hollowY = smilesDict[substrate]
            hollowDict[substrate] = [hollowX , hollowY]
            to_remove.append(substrate)
        
    

    # Remove keys after iteration to avoid modifying the dictionary while looping
    for key in to_remove:
        del smilesDict[key]

    return highlightDict, hollowDict , smilesDict


def makePlots(pcaDict ,  partitionList , chemistryDicts  ,chemistryStr , colors , partitionStr):
    
    for partition in partitionList:
        #print("partition" , partition)
        for i , chemistryDict in enumerate(chemistryDicts): #this is the highlighted chemistry in the plot
            highlightDict , hollowDict , smilesDict = findHighlights(pcaDict.copy()  ,  partition , chemistryDict )


            if len(highlightDict) != 0:
                xHigh , yHigh = zip (*highlightDict.values())
                colorScatter = True
            else:
                colorScatter = False

        
            if len(hollowDict) != 0:
                xHollow , yHollow = zip (*hollowDict.values())
                hollowScatter = True
            else:
                hollowScatter = False


            xBland , yBland = zip(*smilesDict.values())
            chemistryLabel = chemistryStr[i]
            #print("**********************" + str(chemistryLabel))

            xLabel = "PCA1"
            yLabel = "PCA2"
            dpi = 300
            plt.figure(figsize=(800 / dpi, 600 / dpi), dpi=dpi)  # 800x600 pixels
            plt.scatter(list(xBland), list(yBland), c="grey", alpha=0.14 , s=10)
            if colorScatter:
                plt.scatter(list(xHigh), list(yHigh), c=colors[i], alpha=0.7 , s=10)
            if hollowScatter:
                plt.scatter(list(xHollow), list(yHollow), facecolors='none', edgecolors=colors[i], alpha=0.7 , s=10)
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
            #print("90" , type(X))

            text = "\n\n\nFeatures drop due to low variance: " + "".join([f'\n\t{label}' for label in dropped_features])
            f.write(text)

            X, feature_labels, drop_group = correlation_analysis(X, feature_labels, threshold=0.95)
            text = "\n\n\nFeatures drop due correlation: (STILL HAS ISSUES) "
            #print("97" , type(X))
            import json
            text += json.dumps(drop_group, indent=4).replace('\n', '\n\t')
            f.write(text)

            X, feature_labels  = spearmanr_correlation(X, feature_labels, threshold=0.95)
            #print("103" , type(X))
    return X , feature_labels
def eliminateNans(df, nanDict):
    allNanRows = set()
    for key in nanDict.keys():
        rows = nanDict[key]
        allNanRows.update(rows)
    
    df = df.drop(list(allNanRows))
    df = df.reset_index(drop=True)
    return df
def transformations(dataframeDirs):
    usualSuspects = ["SMILES" , "Compound_Name", "Yield", "ChemistryType"  ]

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
if __name__ == "__main__":

    chemistryDirs = str(sys.argv[1])
    datasetDir = str(sys.argv[2])
    chemType = str(sys.argv[3])
    specialK = int(sys.argv[4]) #Kmeans cluster
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
    #print(Xdataframe.columns)
    smileList = Xdataframe["SMILES"].copy()
    Xdataframe = Xdataframe.drop("SMILES", axis=1)


    featureLabels = list(Xdataframe.columns)
    
    X , featureLabels  = featureFiltering(datasetDir , Xdataframe ,featureLabels , chemType)



    pcaDict , clusterList = dimensionalityReduction(X , smileList)
    smilesTot = list(pcaDict.keys())
    coordinateTot = list(pcaDict.values())
    if os.path.exists("pcaCoordinates.dat"):
        for i in range(len(smilesTot)):
        
            smiles = smilesTot[i]
            xAxis = coordinateTot[i][0]
            yAxis = coordinateTot[i][1]
            with open("pcaCoordinates.dat", "a") as file:
                file.write(f"{smiles},{xAxis} ,  {yAxis}\n") 

    chemSpaceDict = []
    chemDirList = []
    for dir in substrateSpaces:
        #print(dir)
        df = pd.read_csv(dir)
        smileList_ = list(df["SMILES"])
        yieldList = list(df["Yield"])
        chemistryDict = dict(zip(smileList_, yieldList))
        chemSpaceDict.append(chemistryDict)
        split1 = dir.split("/")[-1]

        split2 = split1.split(chemType)[0]
        chemDirList.append(split2)
    makePlots(pcaDict.copy() ,  partitionList , chemSpaceDict  , chemDirList, colorList , "Yield")
    
    #recreate X with SMILES and with cluster assignments for featureDistributionGenerator.py 
    X["SMILES"] = smileList
    X["clusterID"] = clusterList
    createCSV(X, datasetDir , chemType + "_PCADataframe")
    





    





