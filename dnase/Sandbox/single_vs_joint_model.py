#############################################################
# Single versus joint model testing.
# Produces figures for paper.
#############################################################


import collections
import datetime

import tensorflow as tf
import h5py
from scipy.io import loadmat
import numpy as np
import sklearn.metrics
import os
import h5sparse
import datetime
import logging
from numba import cuda
from scipy import stats

import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

from scipy.sparse import coo_matrix, vstack

from scipy.fftpack import fft, ifft
from tensorflow.python.client import device_lib
import sys

import json


# # Define Paths for this user

# In[3]:


feature_path = '../data/feature_name'

output_path = '/home/eecs/akmorrow/epitome/out/Epitome'

data_path = "/data/akmorrow/epitome_data/deepsea_labels_train/"


from epitome.functions import *
from epitome.models import *
from epitome.generators import *
from epitome.constants import *



# ### Load DeepSEA data

train_data, valid_data, test_data = load_deepsea_label_data(data_path)


# In[7]:


print(valid_data["y"].shape, train_data["y"].shape, test_data["y"].shape)




# Available cell types
validation_celltypes = ["K562"] # we remove hepg2 from the validation, as there are so few SMC3 cell types to begin with 
test_celltypes = ["HepG2"]


matrix, cellmap, assaymap = get_assays_from_feature_file(feature_path='../../data/feature_name', 
                                  eligible_assays = None,
                                  eligible_cells = None, min_cells_per_assay = 2, min_assays_per_cell=5)
    
inv_assaymap = {v: k for k, v in assaymap.items()}

fig = plt.figure(figsize = (20,10))
ax = fig.add_subplot(1,1,1)
ax.set_aspect('equal')
plt.xticks(np.arange(len(assaymap)), rotation = 90)
ax.set_xticklabels(assaymap.keys())
plt.yticks(np.arange(len(cellmap)))
ax.set_yticklabels(cellmap.keys())


print("training model")
radii = [1,3,10,30]
model = MLP(4, [100, 100, 100, 50], 
            tf.tanh, 
            train_data, 
            valid_data, 
            test_data, 
            test_celltypes,
            gen_from_peaks, 
            matrix,
            assaymap,
            cellmap,
            shuffle_size=2, 
            batch_size=64,
            radii=radii)
model.train(5000)
print("evaluating joint model")

eval_count = 50000
test_DNase = model.test(eval_count, log=True)
model.close()


time = datetime.datetime.now().time().strftime("%Y-%m-%d_%H:%M:%S")

file = open('/data/akmorrow/epitome_data/out/tmp_prediction_aucs_allTFs_%s.json' % time, 'w')

file.write(json.dumps(test_DNase[2]))
file.write("\n")

# flush to file
file.flush()
file.close()



factors = list(assaymap)[1:]
eligible_cells = list(cellmap)
NODATA_PLACEHOLDER = {'AUC': np.NAN,'auPRC': np.NAN,'GINI': np.NAN}


file = open('/data/akmorrow/epitome_data/out/tmp_prediction_aucs_singleTFs_%s.json' % time, 'w')


time = datetime.datetime.now().time().strftime("%Y-%m-%d_%H:%M:%S")

file_single = open('/data/akmorrow/epitome_data/out/tmp_prediction_aucs_singleTFs_%s.json' % time, 'w')

# write opening brace for valid json
file_single.write("[")

for assay in factors:

    print(" Running on assay %s..." % assay)

    label_assays = ['DNase', assay]
    
    try: 
        matrix, cellmap, assaymap = get_assays_from_feature_file(feature_path='../../data/feature_name', eligible_assays = ["DNase", assay], 
                                     eligible_cells = eligible_cells)
        print(cellmap)
        print(test_celltypes)
        

        model = MLP(4, [100, 100, 100, 50], 
                    tf.tanh, 
                    train_data, 
                    valid_data, 
                    test_data, 
                    test_celltypes,
                    gen_from_peaks, 
                    matrix,
                    assaymap,
                    cellmap,
                    shuffle_size=2, 
                    radii=radii)
        model.train(5000)

        test_DNase_1 = model.test(455024, log=True)
        model.close()
        single_results[assay] = test_DNase_1

        file_single.write(json.dumps(test_DNase_1[2]))
        file_single.write(",")

        # flush to file
        file_single.flush()
    except:
        print("%s failed\n" % assay)
        tmp_object = {assay: NODATA_PLACEHOLDER}
        file_single.write(json.dumps(tmp_object))
        
# write closing brace for valid json
file_single.write("]")



# flush to file
file_single.flush()
file_single.close()

##################### PLOTTING #######################

# tmp hack for missing json in singles file, in the case that not all single models get written.
single_names = single_results.keys()
joint_names = joint_results.keys()
single_names

for i in joint_names:
    if i not in single_names:
        single_results[i] = NODATA_PLACEHOLDER
        
# function for jointly plotting 
def joint_plot(dataframe, outlier_filter = "joint < 0"):
    """
    Returns seaborn joint plot 
    
    :param: dataframe of single, joint, and index is TF name
    :outlier_filter: string filter to label. Defaults to no labels.
    """

    ax = sns.jointplot("single", "joint", data=dataframe, kind="reg", color='k', stat_func=None)
    ax.ax_joint.cla()
    # filter for labels
    label_df = df.query(outlier_filter)

    def ann(row):
        ind = row[0]
        r = row[1]
        plt.gca().annotate(ind, xy=(r["single"], r["joint"]), 
                xytext=(2,2) , textcoords ="offset points", )

    for index, row in label_df.iterrows():
        if (not np.isnan(row["single"]) and not np.isnan(row["joint"])):
            ann((index, row))
    
    for i,row in dataframe.iterrows():
        color = "blue" if row["single"] <  row["joint"] else "red"
        ax.ax_joint.plot(row["single"], row["joint"], color=color, marker='o')
        
    return ax

########### ROC Scatter Plot ##############
joint_ROC_values = list(map(lambda x: x["AUC"], joint_results.values()))
single_ROC_values = list(map(lambda x: x["AUC"], single_results.values()))
names = list(joint_names)

d = {'joint': joint_ROC_values, 'single': single_ROC_values}
df = pd.DataFrame(data=d, index=names)


ax = joint_plot(df, "joint < 0.6")
ax.savefig("/home/eecs/akmorrow/epitome/out/figures/Figure_epitome_internals/joint_v_single_auROC.pdf")

########### PR Scatter Plot ##############
joint_auPRC_values = list(map(lambda x: x["auPRC"], joint_results.values()))
single_auPRC_values = list(map(lambda x: x["auPRC"], single_results.values()))

d = {'joint': joint_auPRC_values, 'single': single_auPRC_values}
df = pd.DataFrame(data=d, index=names)


# ax = scatter_plot(single_ROC_values, joint_ROC_values, names, 'Single auROC values', 'Joint auROC values')
ax = joint_plot(df, "joint < 0.1")
ax.savefig("/home/eecs/akmorrow/epitome/out/figures/Figure_epitome_internals/joint_v_single_auPR.pdf")


########### GINI Scatter Plot ##############
joint_auPRC_values = list(map(lambda x: x["GINI"], joint_results.values()))
single_auPRC_values = list(map(lambda x: x["GINI"], single_results.values()))

d = {'joint': joint_auPRC_values, 'single': single_auPRC_values}
df = pd.DataFrame(data=d, index=names)


# ax = scatter_plot(single_ROC_values, joint_ROC_values, names, 'Single auROC values', 'Joint auROC values')
ax = joint_plot(df, "joint < 0.4")
ax.savefig("/home/eecs/akmorrow/epitome/out/figures/Figure_epitome_internals/joint_v_single_GINI.pdf")
