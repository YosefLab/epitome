
# # Runs and evaluates Deepsea

import collections
import os

import tensorflow as tf
import h5py
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import kipoi
from sklearn.metrics import average_precision_score
import argparse

# Absolute path of Metrics folder
current_dirname = os.path.dirname(os.path.abspath(__file__)) # in Metrics

from epitome.functions import *
from epitome.constants import *

feature_path = os.path.join(current_dirname, '../../data/feature_name')

# Parser for user specific locations
parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--deepsea_path', help='deepsea_train data downloaded from DeepSEA (./bin/download_deepsea_data.sh)')
parser.add_argument('--output_path', help='path to save DeepSEA performance pickle files to')

deepsea_path = parser.parse_args().deepsea_path
output_path = parser.parse_args().output_path

if (deepsea_path == None or output_path == None):
    raise ValueError("Invalid inputs for deepsea_path or output_path")

if (not os.path.isdir(output_path)):
    print("%s not found. Creating directory..." % output_path)
    os.mkdir(output_path)


######################### FUNCTIONS ##############################

def deepSEA_performance(data, preds):
    
    df_AUC = pd.DataFrame([],
                       columns= ["CellType"] +assaylist + ["AUC_Average_Macro", "AUC_Average_Micro"])
    df_PR = pd.DataFrame([],
                       columns= ["CellType"] +assaylist + ["PR_Average"])
    
    # holds DeepSEA averaged values (averaged from other cell types)
    df_AV_AUC = pd.DataFrame([],
                       columns= ["CellType"] + assaylist + ["AUC_Average_Macro", "AUC_Average_Micro"])
    df_AV_PR = pd.DataFrame([],
                       columns= ["CellType"] + assaylist + ["PR_Average"])
    
    for celltype in list(cellmap):

        results_AUC, results_PR, results_AV_AUC, results_AV_PR = deepSEA_performance_for_celltype(celltype, data, preds)
        
        df_AUC = df_AUC.append(results_AUC)
        df_PR  = df_PR.append(results_PR)
        
        df_AV_AUC = df_AV_AUC.append(results_AV_AUC)
        df_AV_PR  = df_AV_PR.append(results_AV_PR)
        
    return df_AUC, df_PR, df_AV_AUC, df_AV_PR

def deepSEA_performance_for_celltype(celltype, data, preds):
    
    print(celltype)
    
    y_indices = matrix[cellmap[celltype]]
    
    # indices for averaging results for celltype
    indices_mat = np.delete(matrix, [cellmap[celltype]], axis=0) 
    
    assaylist= list(assaymap)

    # get names assays that exist for this cell type
    cell_assays = list(map(lambda x: x[0], filter(lambda x: x[1] != -1, zip(assaylist, y_indices))))
    cell_preds = preds.T[y_indices[y_indices != -1]]
    cell_truth = data["y"][y_indices[y_indices != -1]]

    results_AUC = [celltype]
    results_PR  = [celltype]
    
    results_AV_AUC = [celltype]
    results_AV_PR  = [celltype]

    # get average predictions for this celltype (takes a while)
    weights = np.tile((indices_mat!=-1).reshape(indices_mat.shape + (1,)), (1, 1, preds.shape[0]))
    tmp = preds.T[indices_mat]
    average_preds = np.average(tmp, axis=0, weights=weights)
    average_preds = average_preds[y_indices != -1]
 
    i = 0
    for assay in assaymap:

        if (assay in cell_assays):
            
            # calculate original values
            try: # catch cases where there is only one truth (0 or 1), which throws a ValueError
                results_AUC.append(sklearn.metrics.roc_auc_score(cell_truth[i].T, cell_preds[i].T, average="macro"))
                results_PR.append(average_precision_score(cell_truth[i].T, cell_preds[i].T))
                
            except ValueError as x:
                        print("%s: %s" % (assay, x))
                        results_AUC.append(np.NAN)
                        results_PR.append(np.NAN)
                        
            # calculate average values
            try: # catch cases where there is only one truth (0 or 1), which throws a ValueError
                results_AV_AUC.append(sklearn.metrics.roc_auc_score(cell_truth[i].T, average_preds[i].T, average="macro"))
                results_AV_PR.append(average_precision_score(cell_truth[i].T, average_preds[i].T))
                
            except ValueError as x:
                        print("%s: %s" % (assay, x))
                        results_AV_AUC.append(np.NAN)
                        results_AV_PR.append(np.NAN)
                        
            i = i + 1
        else:
            results_AUC.append(np.NAN)
            results_PR.append(np.NAN)
            
            results_AV_AUC.append(np.NAN)
            results_AV_PR.append(np.NAN)

    # get total averages for orignal values
    try: # catch cases where there is only one truth (0 or 1), which throws a ValueError
        results_AUC.append(sklearn.metrics.roc_auc_score(cell_truth.T, cell_preds.T, average='macro'))
        results_AUC.append(sklearn.metrics.roc_auc_score(cell_truth.T, cell_preds.T, average='micro'))
        results_PR.append(average_precision_score(cell_truth.T, cell_preds.T))                     
    except ValueError as x:
        print("%s: %s" % (assay, x))
        results_AUC.append(np.NAN)
        results_AUC.append(np.NAN)
        results_PR.append(np.NAN)

        
    # get total averages for averaged values
    try: # catch cases where there is only one truth (0 or 1), which throws a ValueError
        results_AV_AUC.append(sklearn.metrics.roc_auc_score(cell_truth.T, average_preds.T, average='macro'))
        results_AV_AUC.append(sklearn.metrics.roc_auc_score(cell_truth.T, average_preds.T, average='micro'))
        results_AV_PR.append(average_precision_score(cell_truth.T, average_preds.T))                     
    except ValueError as x:
        print("%s: %s" % (assay, x))
        results_AV_AUC.append(np.NAN)
        results_AV_AUC.append(np.NAN)
        results_AV_PR.append(np.NAN)
        
        
    # normal results
    df_AUC = pd.DataFrame([results_AUC],
                       columns=["CellType"] + assaylist + ["AUC_Average_Macro", "AUC_Average_Micro"])
    df_PR = pd.DataFrame([results_PR],
                       columns=["CellType"] + assaylist + ["PR_Average"])
    
    # averaged results
    df_AV_AUC = pd.DataFrame([results_AV_AUC],
                       columns=["CellType"] + assaylist + ["AUC_Average_Macro", "AUC_Average_Micro"])
    df_AV_PR = pd.DataFrame([results_AV_PR],
                       columns=["CellType"] + assaylist + ["PR_Average"])
        
        
    return df_AUC, df_PR, df_AV_AUC, df_AV_PR

######################### END FUNCTIONS ##############################

# # Load test Data
_, _, test_data = load_deepsea_data(deepsea_path)
print(test_data["x"].shape)
print(test_data["y"].shape)

# # Choose cell types and assays to validate on


# matrix is cell types by factors, contains indices in feature vector
# 172 factors.
matrix, cellmap, assaymap = get_assays_from_feature_file(feature_path=feature_path,eligible_assays = None,
                                  eligible_cells = None, min_cells_per_assay = 2, min_assays_per_cell=2)

assaylist= list(assaymap)

# # Check how DeepSea does

model = kipoi.get_model('DeepSEA/predict') # load the model

print("loaded model")

batch_size = 100
preds = []
for i in np.arange(0, len(test_data["x"]), batch_size):
    print(i)
    batch = test_data["x"][i:i+batch_size]
    batch = np.expand_dims(batch, 2)
    batch = batch[:,[0,2,1,3]]
    preds.append(model.predict_on_batch(batch.astype(np.float32)))
preds = np.concatenate(preds, axis=0)

print("finished predictions")

# ### Evaluate all factors and save results

import warnings
warnings.filterwarnings('ignore')

df_AUC, df_PR, df_AV_AUC, df_AV_PR = deepSEA_performance(test_data, preds)

print("Saving results to %s" % output_path)

# can be read back in using pd.read_csv
df_AUC.to_csv(os.path.join(output_path, "DeepSEA_AUC.csv"), sep='\t')

df_PR.to_csv(os.path.join(output_path, "DeepSEA_PR.csv"), sep='\t')

df_AV_AUC.to_csv(os.path.join(output_path, "DeepSEA_AVERAGE_AUC.csv"), sep='\t')

df_AV_PR.to_csv(os.path.join(output_path, "DeepSEA_AVERAGE_PR.csv"), sep='\t')


