

############# Imports ###########################
import collections
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import os
import torch
import h5sparse
import datetime
import logging

from scipy import stats

import datetime
import sys
import argparse

from epitome.functions import *
from epitome.constants import *
from epitome.generators import *
from epitome.models import *


############## Read in user params ################

# Parser for user specific locations
parser = argparse.ArgumentParser(description='run Epitome metrics')

parser.add_argument('--feature_path', help='deepsea_train data downloaded from DeepSEA (./bin/download_deepsea_data.sh)')
parser.add_argument('--model_path', help='path to load Epitome model from. Saved using model.save()')
parser.add_argument('--data_path', help='deepsea_train label data downloaded from DeepSEA (./bin/download_deepsea_data.sh) and processed with  save_deepsea_label_data() in functions.py')
parser.add_argument('--output_path', help='path to save Epitome performance pickle files to')

feature_path = parser.parse_args().feature_path
model_path = parser.parse_args().model_path
data_path = parser.parse_args().data_path
output_path = parser.parse_args().output_path

if (not os.path.isfile(feature_path)):
    raise ValueError("Invalid feature_path %s" % feature_path)

# make sure checkpoint index file exists
if (not os.path.isfile(model_path + ".index")):
    raise ValueError("Invalid model path %s" % model_path)
    
if (not os.path.isdir(data_path)):
    raise ValueError("Invalid data path %s" % data_path)
    
if (not os.path.isdir(output_path)):
    print("%s not found. Creating directory..." % output_path)
    os.mkdir(output_path)


############## Load Data ########################

# generated from original data by save_deepsea_label_data(deepsea_path) in functions.py
train_data, valid_data, test_data = load_deepsea_label_data(data_path)

matrix, cellmap, assaymap = get_assays_from_feature_file(feature_path=feature_path, 
                                  eligible_assays = None,
                                  eligible_cells = None, min_cells_per_assay = 2, min_assays_per_cell=5)

############## Load Model #######################

model  = MLP(4, [100, 100, 100, 50], 
            tf.tanh, 
            train_data, 
            valid_data, 
            test_data, 
            [],
            gen_from_peaks, 
            matrix,
            assaymap,
            cellmap,
            shuffle_size=2, 
            radii=[1,3,10,30])

model.restore(model_path)


############# Run Evaluation ####################

# load in all cell types for evaluation (64 cell types)
all_matrix, all_cellmap, all_assaymap = get_assays_from_feature_file(feature_path=feature_path,eligible_assays = list(assaymap),
                                  eligible_cells = None, min_cells_per_assay = 2, min_assays_per_cell=2)


assaylist = list(assaymap)
assaylist.remove('DNase')

df_AUC = pd.DataFrame([],
                   columns= ["CellType"] + assaylist + ["AUC_Average_Macro", "AUC_Average_Micro"])
df_PR = pd.DataFrame([],
                   columns= ["CellType"] + assaylist + ["PR_Average"])
    
for test_celltype in all_cellmap:
    print(test_celltype)
    
    eval_cell_types = list(cellmap).copy()
    
    # if test_celltype is in eval_cell_types, replace it with something else
    if (test_celltype in eval_cell_types):
        if (test_celltype == "PANC-1"):
            new_eval_celltype = "NT2-D1" # TODO AM 4/1/2019 maybe don't hardcode
        else:
            new_eval_celltype = "PANC-1" # TODO AM 4/1/2019 maybe don't hardcode
        
        print("removing %s from eval_celltypes and replacing with %s" % (test_celltype, new_eval_celltype))
        eval_cell_types.remove(test_celltype)
        eval_cell_types.append(new_eval_celltype)
    
    _, iter_ = generator_to_one_shot_iterator(make_dataset(test_data, 
                                                   [test_celltype], 
                                                   eval_cell_types,
                                                   gen_from_peaks, 
                                                   all_matrix,
                                                   assaymap,
                                                   all_cellmap,
                                                   radii = model.radii, mode = Dataset.TEST),
                                                       model.batch_size, 1, model.prefetch_size)
                
    
    preds, truth, assay_dict, microAUC, macroAUC, _ = model.test_from_generator(test_data["y"].shape[1], iter_, log=True)
    
    parsed_AUC = list(map(lambda x: x[1]["AUC"], assay_dict.items()))
    parsed_PR = list(map(lambda x: x[1]["auPRC"], assay_dict.items()))
    
    # get auPRC mean
    average_auPRC = np.nanmean(parsed_PR)

    results_AUC = pd.DataFrame([[test_celltype] + parsed_AUC +  [microAUC, macroAUC]],
                       columns=["CellType"] + assaylist + ["AUC_Average_Macro", "AUC_Average_Micro"])
    
    results_PR = pd.DataFrame([[test_celltype] + parsed_PR +  [average_auPRC]],
                       columns=["CellType"] + assaylist + ["PR_Average"])
    
    
    df_AUC = df_AUC.append(results_AUC)
    df_PR  = df_PR.append(results_PR)
    
    
    
print("saving csv files to %s" % output_path)
df_AUC.to_csv(os.path.join(output_path, "Epitome_AUC.csv"),sep='\t')

df_PR.to_csv(os.path.join(output_path, "Epitome_PR.csv"), sep='\t')