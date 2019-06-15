
# This file takes as input a directory of a single bed file
# and uses Epitome to predict results for each TF at each
# position in the bed file.


############## Imports #################

import collections
 
import tensorflow as tf
import h5py
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import os
import pybedtools
import torch
import h5sparse
import datetime
import logging

import tempfile
from scipy import stats
import argparse
import sys

# Absolute path of Metrics folder
current_dirname = os.path.dirname(os.path.abspath(__file__)) # in Metrics

# Files in repository
from epitome.functions import *
from epitome.models import *
from epitome.generators import *
from epitome.constants import *

########################## PARSE USER ARGUMENTS ###########################

# Parser for user specific locations
parser = argparse.ArgumentParser(description='Runs Epitome on a directory of chromatin bed files.')

parser.add_argument('--deepsea_path', help='deepsea_train data downloaded from DeepSEA (./bin/download_deepsea_data.sh)')
parser.add_argument('--label_path', help='deepsea_train label data downloaded and processed from DeepSEA (./bin/download_deepsea_data.sh)')
parser.add_argument('--model_path', help='path to load model from') 
parser.add_argument('--bed_file', help='path single bed file')
parser.add_argument('--output_path', help='path to save tsv of results to')

deepsea_path = parser.parse_args().deepsea_path
output_path = parser.parse_args().output_path
model_path = parser.parse_args().model_path
peak_file = parser.parse_args().bed_file
label_path = parser.parse_args().label_path
     

if (deepsea_path == None or output_path == None or model_path == None or peak_file == None or label_path == None):
    raise ValueError("Invalid inputs %s" % parser)

if (not os.path.isfile(peak_file)):
     raise ValueError("%s is not a valid file" % (peak_file))

if (not os.path.isdir(deepsea_path)):
     raise ValueError("%s is not a valid data directory" % (deepsea_path))
        
if (not os.path.isdir(label_path)):
     raise ValueError("%s is not a valid data directory" % (label_path))

# make sure checkpoint index file exists
if (not os.path.isfile(model_path + ".index")):
    raise ValueError("Invalid model path %s" % model_path)



#################### Start code ##############################
train_data, valid_data, test_data = load_deepsea_label_data(label_path)
data = {Dataset.TRAIN: train_data, Dataset.VALID: valid_data, Dataset.TEST: test_data}   

matrix, cellmap, assaymap = get_assays_from_feature_file(eligible_assays = None,
                                  eligible_cells = None, 
                                  min_cells_per_assay = 2, 
                                  min_assays_per_cell=5)
     
     
############## Load Model #######################

model  = MLP(4, [100, 100, 100, 50], 
            tf.tanh, 
            data,
            [],
            gen_from_peaks, 
            matrix,
            assaymap,
            cellmap,
            shuffle_size=2, 
            radii=[1,3,10,30])

model.restore(model_path)
     
     
############## score bed file ###################
     
results = model.score_peak_file(peak_file)

print("writing results to %s" % output_path)
# save final results
results.to_csv(output_path, sep='\t')

