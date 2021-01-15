# ## Download all data for ChIP-Atlas for a specified assembly

############################## Imports ####################################

import pandas as pd
import numpy as np
import os
import urllib
import multiprocessing as mp
from multiprocessing import Pool
import subprocess
import math
import argparse
import h5py
import re
from itertools import islice
import scipy.sparse
from epitome.functions import *
import sys
import shutil
import logging
import traceback
import glob

from download_functions import *

##################################### LOG INFO ################################
logger = set_logger()

##############################################################################################
############################################# PARSE USER ARGUMENTS ###########################
##############################################################################################

# Parser for user specific locations
parser = argparse.ArgumentParser(description='Downloads ChIP-Atlas data from a chip_atlas_experiment_list.csv file.')

parser.add_argument('download_path', help='Temporary path to download bed/bigbed files to.', type=str)
parser.add_argument('assembly', help='assembly to filter files in metadata.tsv file by.', choices=['ce10', 'ce11', 'dm3', 'dm6', 'hg19', 'hg38', 'mm10', 'mm9', 'rn6', 'sacCer3'], type=str)
parser.add_argument('output_path', help='path to save file data to', type=str)

parser.add_argument('--metadata_url',type=str, default="ftp://ftp.biosciencedbc.jp/archive/chip-atlas/LATEST/chip_atlas_experiment_list.zip",
                    help='ChIP-Atlas metadata URL.')

parser.add_argument('--min_chip_per_cell', help='Minimum ChIP-seq experiments for each cell type.', type=int, default=1)
parser.add_argument('--min_cells_per_chip', help='Minimum cells a given ChIP-seq target must be observed in.', type=int, default=3)

parser.add_argument('--regions_file', help='File to read regions from', type=str, default=None)


download_path = parser.parse_args().download_path
assembly = parser.parse_args().assembly
output_path = parser.parse_args().output_path
metadata_path = parser.parse_args().metadata_url
min_chip_per_cell = parser.parse_args().min_chip_per_cell
min_cells_per_chip = parser.parse_args().min_cells_per_chip
all_regions_file_unfiltered = parser.parse_args().regions_file

# append assembly to all output paths so you don't overwrite previous runs
meta_download_path = download_path
download_path = os.path.join(download_path, assembly)
output_path = os.path.join(output_path, assembly)

# where to temporarily store np files
tmp_download_path = os.path.join(download_path, "tmp_np")
bed_download_path = os.path.join(download_path, "downloads")

# TODO RM
assert len(glob.glob(os.path.join(tmp_download_path,'*')))>1000
assert len(glob.glob(os.path.join(bed_download_path,'*.bed')))>1000

# make paths if they do not exist
if not os.path.exists(download_path):
    os.makedirs(download_path)
if not os.path.exists(output_path):
    os.makedirs(output_path)
if not os.path.exists(tmp_download_path):
    os.makedirs(tmp_download_path)
if not os.path.exists(bed_download_path):
    os.makedirs(bed_download_path)

if all_regions_file_unfiltered is None:
    # path to save regions to. must be defined before loj_overlap function
    all_regions_file_unfiltered = os.path.join(output_path,"all.pos_unfiltered.bed")
else:
    # copy all regions file to output path if not already there
    if os.path.normpath(os.path.dirf(all_regions_file_unfiltered)) != os.path.normpath(output_path):
        shutil.copyfile(all_regions_file_unfiltered, os.path.join(output_path, "all.pos_unfiltered.bed"))


# download metadata if it does not exist
metadata_file = os.path.join(meta_download_path, os.path.basename(metadata_path).replace('.zip','.csv'))

if not os.path.exists(metadata_file):
    zipped = os.path.join(download_path, os.path.basename(metadata_path))

    if not os.path.exists(zipped):
        subprocess.check_call(["wget", "-O", zipped, "-np", "-r", "-nd", metadata_path])

    # gunzip file
    subprocess.check_call(["unzip", zipped])

# files = pd.read_csv(metadata_file, engine='python') # needed for decoding
#
# ##############################################################################################
# ######### get all files that are peak files for histone marks or TF ChiP-seq #################
# ##############################################################################################
#
# # assembly column is either 'Assembly' or 'File assembly'
# assembly_column = files.filter(regex=re.compile('Assembly', re.IGNORECASE)).columns[0]
#
# antigen_classes = ['DNase-seq','Histone','TFs and others']
#
# assembly_files = files[(files[assembly_column] == assembly) &
#                                    (files['Antigen class'].isin(antigen_classes))]
#
# # Get unique by Antigen class, Antigen, Cell type class, Cell type, Cell type description.
# rm_dups = assembly_files[['Antigen class', 'Antigen', 'Cell type class', 'Cell type', 'Cell type description']].drop_duplicates()
# filtered_files = assembly_files.loc[rm_dups.index]
#
# # get unique dnase experiments
# filtered_dnase = filtered_files[((filtered_files["Antigen class"] == "DNase-seq"))]
#
# chip_files = filtered_files[(((filtered_files["Antigen class"] == 'Histone') | (filtered_files["Antigen class"] == 'TFs and others')))]
#
# # only want ChIP-seq from cell lines that have DNase
# filtered_chip = chip_files[(chip_files["Cell type"].isin(filtered_dnase["Cell type"]))]
#
# # only want assays that are shared between more than 3 cells
# filtered_chip = filtered_chip.groupby("Antigen").filter(lambda x: len(x) >= min_cells_per_chip)
#
# # only want cells that have more than min_chip_per_cell epigenetic marks
# filtered_chip = filtered_chip.groupby("Cell type").filter(lambda x: len(x) >= min_chip_per_cell)
#
# # only filter if use requires at least one chip experiment for a cell type.
# if min_chip_per_cell > 0:
#     # only want DNase that has chip.
#     filtered_dnase = filtered_dnase[(filtered_dnase["Cell type"].isin(filtered_chip["Cell type"]))]
#
# # combine dataframes
# filtered_files = filtered_dnase.append(filtered_chip)
# filtered_files.reset_index(inplace = True)
#
# # group by antigen/celltype combinations. Iterate over these
# replicate_groups = assembly_files[(assembly_files['Antigen'].isin(filtered_files['Antigen'])) &
#                                   (assembly_files['Cell type'].isin(filtered_files['Cell type']))]
#
# # read in annotated Antigens
# this_dir = os.path.dirname(os.path.abspath(__file__))
# TF_categories = pd.read_csv(os.path.join(this_dir,'ChIP_target_types.csv'),sep=',')
# TF_categories.replace({'DNase': 'DNase-Seq'}, inplace=True)
#
# # sanity check that all antigens are accounted for in TF_categories
# assert len([i for i in set(replicate_groups['Antigen']) if i not in list(TF_categories['Name'])]) == 0
#
# # Filter out ChIP-seq not in TFs, accessibility, histones, etc. We lose about 1100 rows
# filtered_names = TF_categories[TF_categories['Group'].isin(['TF','chromatin accessibility','chromatin modifier','histone',
#  'histone modification'])]
#
# replicate_groups = replicate_groups[replicate_groups['Antigen'].isin(filtered_names['Name'])]
# replicate_groups.reset_index(inplace = True)
#
# logger.info("Processing %i antigens and %i experiments" % (len(set(replicate_groups['Antigen'])), len(replicate_groups)))
#
# # group experiments together
# replicate_groups = replicate_groups.groupby(['Antigen', 'Cell type'])

##############################################################################################
############################# window chromsizes into 200bp ###################################
##############################################################################################
window_genome(all_regions_file_unfiltered,
                download_path,
                assembly)

#############################################################################################
################################ save all files to matrix ###################################
#############################################################################################
# call parallel download code
script = os.path.join(this_dir, 'parallel_download.py')
parallel_cmd = [script, download_path, assembly,
                        '--metadata_path', metadata_path,
                        '--min_chip_per_cell',min_chip_per_cell,
                        '--min_cells_per_chip', min_cells_per_chip,
                        '--all_regions_file',all_regions_file_unfiltered
                        ]

process = subprocess.Popen(parallel_cmd, stdout=subprocess.PIPE)
stdout = process.communicate()[0]

while True:
    output = process.stdout.readline()
    if output == '' and process.poll() is not None:
        break
    if output:
        print output.strip()
rc = process.poll()

# create matrix or load in existing
matrix_path_all = os.path.join(download_path, 'train_total.h5') # all sites

# written in parralel_download.py
row_df = pd.read_csv(os.path.join(download_path, "row_df.csv"))
#
# # collect all regions and merge by chromsome, count number of 200bp bins
# pyDF = pr.read_bed(all_regions_file_unfiltered)
#
# tmp = list(replicate_groups)
# with Pool(threads) as p:
#     # list of tuples for each file, where tuple is (i, filename, featurename)
#     results = p.starmap(processGroups, list(zip( tmp, [tmp_download_path]* len(tmp), [bed_download_path]* len(tmp) )))
#
# results = [i for i in results if i is not None]
#
# # load in cells and targets into a dataframe
# cellTypes = [i[1] for i in results]
# targets = [i[2] for i in results]
# row_df = pd.DataFrame({'cellType': cellTypes,'target': targets})
#
# ### save matrix
# if os.path.exists(matrix_path_all):
#
#     # make sure the dataset hasnt changed if you are appending
#     assert(matrix[0,:].shape[0] == nregions)
#     assert(matrix[:,0].shape[0] == len(results))
#
# else:
#     h5_file = h5py.File(matrix_path_all, "w")
#     matrix = h5_file.create_dataset("data", (len(results), nregions), dtype='i1', # int8
#         compression='gzip', compression_opts=9)
#
#     for i, (f, cell, target) in enumerate(results):
#
#         matrix[i,:] = np.load(f + ".npz", allow_pickle=True)['data'].astype('i1') # int8
#
#         if i % 100 == 0:
#             logger.info("Writing %i, feature %s..." % (i, feature_name))
#
#     h5_file.close()

logger.info("Done saving sparse data")

# finally, save outputs
save_epitome_dataset(download_path,
                        output_path,
                        matrix_path_all,
                        all_regions_file_unfiltered,
                        row_df,
                        assembly,
                        "CHIPATLAS")

# rm tmp unfiltered bed files
os.remove(all_regions_file_unfiltered)
os.remove(all_regions_file_unfiltered + ".tmp")
# remove h5 file with all zeros
os.remove(matrix_path_all) # remove h5 file with all zeros
