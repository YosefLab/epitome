
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
exec(open(os.path.join(current_dirname, "./constants.py")).read())
exec(open(os.path.join(current_dirname, "./functions.py")).read())
exec(open(os.path.join(current_dirname, "./generators.py")).read())
exec(open(os.path.join(current_dirname, "./models.py")).read())
feature_path = os.path.join(current_dirname, '../data/feature_name')

_DEEPSEA_GENOME_REGIONS_FILENAME = os.path.join(current_dirname,'../data/allTFs.pos.bed')


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

#################### Functions ###############################
     
def score_peak_file(peak_file):
    
    print(peak_file)
    f = peak_file.split("/")[-1]
    
    # get peak_vector, which is a vector matching train set. Some peaks will not overlap train set, 
    # and are stored in pos_windows for future use.
    # This is taking hours?
    peak_vector, pos_windows = bedFile2Vector(peak_file, _DEEPSEA_GENOME_REGIONS_FILENAME, duplicate=False)
    print("finished loading peak file")
    
    # only select peaks to score
    idx = np.where(peak_vector == 1)[0]

    # takes about 1.5 minutes for 100,000 regions TODO AM 4/3/2019 speed up generator
    predictions = model.eval_vector(all_data, peak_vector, idx)

    # map predictions with idx
    zipped = list(zip(idx, predictions))
    # map zipped predictions to positions
    
    def bedRow2Interval(x):
        """
        Converts single row from pybedtools.BedTool to pybedtools.Interval
        
        :param x: row from Bedtools
        
        returns:
            pybedtools.Interval
        """
        return pybedtools.Interval(x.chrom, x.start, x.end)
    
    positions = BedTool(_DEEPSEA_GENOME_REGIONS_FILENAME)
    # map all predictions to key, value of positions bedrow and predictions
    mapped = dict(map(lambda x: (bedRow2Interval(positions[int(x[0])]), x[1]), zipped))
    
    def checkInterval(x, default_factor_count):
        """
        Checks if interval is in mapped. If present returns
        predictions for that interval. Otherwise, returns np zeros.
        
        :param x: row from Bedtools
        
        returns:
            tupple of peak interval from bed file and predictions
        """
        chrom = x[6]
        start = int(x[7])
        end = int(x[8])
        
        peak_interval = str(pybedtools.Interval(x[0], int(x[1]), int(x[2])))

        if start > 0 and pybedtools.Interval(chrom, start, end) in mapped.keys():
            return(peak_interval, mapped[pybedtools.Interval(chrom, start, end)])

        else:
            return(peak_interval, np.zeros(default_factor_count)) 

    # get number of factors to fill in if values are missing
    num_factors = list(mapped.values())[0].shape[0]
    
    # map of peaks and predictions. If peak did not match with training positions
    # returns all 0's
    result = map(lambda x: checkInterval(x, num_factors), pos_windows)
    
    def reduceMeans(kv):
        """
        Takes a grouped set of keys and itergroup of np arrays and
        reduces the group by means. For the purpose if there are 
        multiple predictions mapping to a single region in the query 
        bed file.
        
        :param kv: single item from groupby command. k is key, v, is group of np arrays
        
        returns:
            meaned np arrays for this group
        """
        key = kv[0]
        group = kv[1]
        res = list(map(lambda x: np.matrix(x[1]), group))
        arr = np.concatenate(list(res), axis = 0)
        return(np.mean(arr, axis = 0))


    # mean np arraays by bed file peak (there may be multiple predictions for each peak)
    grouped = list(map(lambda kv: reduceMeans(kv), groupby( result, lambda x: x[0])))
    # join predictions to single matrix

    final = np.concatenate(grouped, axis=0)

    df = pd.DataFrame(final, columns=list(assaymap)[1:])

    # load in peaks to get positions and could be called only once
    df_pos = BedTool(peak_file).to_dataframe()[["chrom", "start", "end"]]
    final_df = pd.concat([df_pos, df], axis=1)
    
    return final_df

     
     
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
     
     
############## score bed file ###################
     
results = score_peak_file(peak_file)

print("writing results to %s" % output_path)
# save final results
results.to_csv(output_path, sep='\t')

