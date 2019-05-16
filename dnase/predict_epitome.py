
# This file takes as input a directory of bed files
# and uses Epitome to predict results. Outputs means.


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
feature_path = os.path.join(current_dirname, '../data/feature_name')

_DEEPSEA_GENOME_REGIONS_FILENAME = os.path.join(current_dirname,'../data/allTFs.pos.bed')


########################## PARSE USER ARGUMENTS ###########################

# Parser for user specific locations
parser = argparse.ArgumentParser(description='Runs Epitome on a directory of chromatin bed files.')

parser.add_argument('--deepsea_path', help='deepsea_train data downloaded from DeepSEA (./bin/download_deepsea_data.sh)')
parser.add_argument('--label_path', help='deepsea_train label data downloaded and processed from DeepSEA (./bin/download_deepsea_data.sh)')
parser.add_argument('--model_path', help='path to load model from') 
parser.add_argument('--bed_files', help='path to directory of bed files')
parser.add_argument('--output_path', help='path to save tsv of results to')

deepsea_path = parser.parse_args().deepsea_path
output_path = parser.parse_args().output_path
model_path = parser.parse_args().model_path
bed_file_dir = parser.parse_args().bed_files
label_path = parser.parse_args().label_path
     

if (deepsea_path == None or output_path == None or model_path == None or bed_file_dir == None or label_path == None):
    raise ValueError("Invalid inputs %s" % parser)

if (not os.path.isdir(bed_file_dir)):
     raise ValueError("%s is not a valid directory with bedfiles" % (bed_file_dir))

if (not os.path.isdir(deepsea_path)):
     raise ValueError("%s is not a valid data directory" % (deepsea_path))
        
if (not os.path.isdir(label_path)):
     raise ValueError("%s is not a valid data directory" % (label_path))

# make sure checkpoint index file exists
if (not os.path.isfile(model_path + ".index")):
    raise ValueError("Invalid model path %s" % model_path)

#################### Functions ###############################
     
def score_peak_file(peak_file):
    print(peak_file)
    f = peak_file.split("/")[-1]
    
    # takes about 20 seconds
    peak_vector = bedFile2Vector(peak_file, _DEEPSEA_GENOME_REGIONS_FILENAME, duplicate=False)

    # only select peaks
    idx = np.where(peak_vector == 1)[0]

    sub_vector = peak_vector[idx]

    sub_data = {
        "y": all_data["y"][:,idx]
    }

    # takes about 1.5 minutes for 100,000 regions TODO AM 4/3/2019 speed up generator
    tmp_results = model.eval_vector(sub_data, sub_vector)
    
    # TODO AM 4/3/2019: taking the mean here is a very naive statistic for measuring binding. 
    # means = np.mean(tmp_results, axis=0)
    
    # sum all TFs 
    sums  = np.sum(tmp_results,axis=0)
    
    return dict(zip(df.columns, [f] + list(sums)))
     
     
#################### Start code ##############################
     
# TODO AM 4/3/2019: make this work with just the labels so it runs faster
all_data = load_deepsea_data_allpos_file(deepsea_path)
train_data, valid_data, test_data = load_deepsea_label_data(label_path)
     
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
     
     
############## Run through all bed files ##########
     
     
df = pd.DataFrame([], columns = ["File"] + list(assaymap)[1:]) # skip DNase
     
# TODO multiprocessing not working here, need a way to speed this up
files = list(map(lambda x: os.path.join(bed_file_dir, x), os.listdir(bed_file_dir)))

bkg = None # holds all peaks for background

for peak_file in files:
    r = score_peak_file(peak_file)
    df = df.append(r, ignore_index=True)
    
    # append peaks to background
    if (bkg == None):
        bkg = BedTool(peak_file)
    else:
        bkg = bkg.cat(BedTool(peak_file))

df.to_csv(output_path + "_tmp_df_results", sep='\t')

# temporarily save background peaks to generate pybedtool from
bkg_tmp_filename = next(tempfile._get_candidate_names())
bkg.saveas(bkg_tmp_filename)

# score background and create df
bkg_score = score_peak_file(bkg_tmp_filename)
bkg_df = pd.DataFrame([], columns = ["File"] + list(assaymap)[1:]) # skip DNase
bkg_df = bkg_df.append(bkg_score, ignore_index=True)

# save bkg temporarily
bkg_df.to_csv(output_path + "_tmp_bkg_results", sep='\t')

# divide each row by background. Take log to smooth out small numbers that generate outliers
k = np.log(df[list(assaymap)[1:]].as_matrix())/np.log(bkg_df[list(assaymap)[1:]].as_matrix())
df[list(assaymap)[1:]] = k

# save final results
df.to_csv(output_path, sep='\t')

# delete temporary files
os.remove(bkg_tmp_filename)
