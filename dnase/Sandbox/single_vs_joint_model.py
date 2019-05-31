#############################################################
# Single versus joint model testing.
# Produces figures for paper.
#############################################################

import collections
import datetime

import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

import tensorflow as tf
import h5py
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics

import os

# from numpy.random import choice

# import h5sparse
import datetime
import logging
from numba import cuda
from scipy import stats

from scipy.sparse import coo_matrix, vstack

from scipy.fftpack import fft, ifft
import sys

import yaml


import json

from epitome.constants import *
from epitome.models import *
from epitome.generators import *
from epitome.functions import *
from epitome.viz import *

# load in user paths
with open('/home/eecs/akmorrow/epitome/config.yml') as f:
    config = yaml.safe_load(f)


# ### Load DeepSEA data
train_data, valid_data, test_data = load_deepsea_label_data(config["data_path"])
data = {Dataset.TRAIN: train_data, Dataset.VALID: valid_data, Dataset.TEST: test_data}

# Available cell types
test_celltypes = ["K562"]

train_iters = 5000
test_iters = 50000


matrix, cellmap, assaymap = get_assays_from_feature_file(feature_path=config['feature_name_file'], 
                                  eligible_assays = None,
                                  eligible_cells = None, min_cells_per_assay = 2, min_assays_per_cell=5)
    

print("training model")
radii = [1,3,10,30]
shuffle_size = 2 # train data indices are already shuffled during sampling

label = "adaptive_sampling"
file_joint_name = '/home/eecs/akmorrow/epitome/out/Epitome/scratch/tmp_prediction_aucs_allTFs_%s.json' % label
                  
model = MLP(4, [100, 100, 100, 50], 
            tf.tanh, 
            data,
            test_celltypes,
            matrix,
            assaymap,
            cellmap,
            shuffle_size=shuffle_size, 
            prefetch_size = 64,
            batch_size=64,
            radii=radii)

# define a set iterator for testing
g = load_data(data[Dataset.VALID], 
                 model.eval_cell_types,  # used for labels. Should be all for train/eval and subset for test
                 model.eval_cell_types,   # used for rotating features. Should be all - test for train/eval
                 matrix,
                 assaymap,
                 cellmap,
                 model.radii, mode = Dataset.VALID)

iter_ = generator_to_one_shot_iterator(g, model.batch_size, 1, model.prefetch_size)
model.train(train_iters)

test_DNase = model.test_from_generator(test_iters, iter_[1], log=True)
model.close()

# write
file_joint = open(file_joint_name, 'w')

file_joint.write(json.dumps(test_DNase[2]))
file_joint.write("\n")

# flush to file
file_joint.flush()
file_joint.close()

### Single models ###

factors = list(assaymap)[1:]
eligible_cells = list(cellmap)
NODATA_PLACEHOLDER = {'AUC': np.NAN,'auPRC': np.NAN,'GINI': np.NAN}

file_single_name = '/home/eecs/akmorrow/epitome/out/Epitome/scratch/prediction_aucs_singleTFs_%s.json' % test_celltypes[0]
file_single = open(file_single_name, 'w')

# write opening brace for valid json
file_single.write("[")

i = 0
for assay in factors:
    
    try: # sometimes not enough positives or negatives 
    
        print(" Running on assay %s..." % assay)

        label_assays = ['DNase', assay]

        matrix, cellmap, assaymap = get_assays_from_feature_file(feature_path=config['feature_name_file'], 
                                                                 eligible_assays = ["DNase", assay], 
                                                                 eligible_cells = eligible_cells)

        model = MLP(4, [100, 100, 100, 50], 
                    tf.tanh,
                    data, 
                    test_celltypes,
                    matrix,
                    assaymap,
                    cellmap,
                    shuffle_size=2, 
                    radii=radii)
        model.train(train_iters)

        

        # define a set iterator for testing
        g = load_data(data[Dataset.VALID], 
                         model.eval_cell_types,  # used for labels. Should be all for train/eval and subset for test
                         model.eval_cell_types,   # used for rotating features. Should be all - test for train/eval
                         matrix,
                         assaymap,
                         cellmap,
                         model.radii, mode = Dataset.VALID)

        iter_ = generator_to_one_shot_iterator(g, model.batch_size, 1, model.prefetch_size)

        test_DNase = model.test_from_generator(test_iters, iter_[1], log=True)

        file_single.write(json.dumps(test_DNase[2]))

        # flush to file
        file_single.flush()
        
    except Exception as e:
        print("%s failed\n with message %s" % (assay, str(e)))
        tmp_object = {assay: NODATA_PLACEHOLDER}
        file_single.write(json.dumps(tmp_object))
        
    if i < (len(factors)-1):
        file_single.write(",")
    i = i + 1
        
# write closing brace for valid json
file_single.write("]")


# flush to file
file_single.flush()
file_single.close()

##################### PLOTTING #######################
#Read JSON data into the datastore variable
with open(file_single_name, 'r') as f:
    single_results_tmp = json.load(f)

# # Flatten nested single results
single_results = {}
for d in single_results_tmp:
    single_results.update(d)
    
with open(file_joint_name, 'r') as f:
    joint_results = json.load(f)

ax = joint_plot(single_results, 
               joint_results, 
               metric = "AUC", 
               model1_name = "single", 
               model2_name = "joint",
               outlier_filter = "joint < single")
ax.savefig("/home/eecs/akmorrow/epitome/out/figures/Figure_epitome_internals/joint_v_single_auROC.pdf")


ax = joint_plot(single_results, 
               joint_results, 
               metric = "auPRC", 
               model1_name = "single", 
               model2_name = "joint",
               outlier_filter = "joint < single")
ax.savefig("/home/eecs/akmorrow/epitome/out/figures/Figure_epitome_internals/joint_v_single_auPRC.pdf")


ax = joint_plot(single_results, 
               joint_results, 
               metric = "GINI", 
               model1_name = "single", 
               model2_name = "joint",
               outlier_filter = "joint < single")
ax.savefig("/home/eecs/akmorrow/epitome/out/figures/Figure_epitome_internals/joint_v_single_GINI.pdf")

