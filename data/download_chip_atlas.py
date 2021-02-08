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

download_path = parser.parse_args().download_path
assembly = parser.parse_args().assembly
output_path = parser.parse_args().output_path
metadata_path = parser.parse_args().metadata_url
min_chip_per_cell = parser.parse_args().min_chip_per_cell
min_cells_per_chip = parser.parse_args().min_cells_per_chip

# append assembly to all output paths so you don't overwrite previous runs
meta_download_path = download_path
download_path = os.path.join(download_path, assembly)
output_path = os.path.join(output_path, assembly)

# make paths if they do not exist
if not os.path.exists(download_path):
    os.makedirs(download_path)
if not os.path.exists(output_path):
    os.makedirs(output_path)

# set file to save regions to
all_regions_file_unfiltered = os.path.join(output_path,"all.pos_unfiltered.bed")

# download metadata if it does not exist
metadata_file = os.path.join(meta_download_path, os.path.basename(metadata_path).replace('.zip','.csv'))

if not os.path.exists(metadata_file):
    zipped = os.path.join(download_path, os.path.basename(metadata_path))

    if not os.path.exists(zipped):
        subprocess.check_call(["wget", "-O", zipped, "-np", "-r", "-nd", metadata_path])

    # gunzip file
    subprocess.check_call(["unzip", zipped])

##############################################################################################
############################# window chromsizes into 200bp ###################################
##############################################################################################
window_genome(all_regions_file_unfiltered,
                download_path,
                assembly)

#############################################################################################
################################### download all files ######################################
#############################################################################################
# call parallel download code
this_dir = os.path.dirname(os.path.abspath(__file__))
script = os.path.join(this_dir, 'parallel_download.py')
parallel_cmd = [sys.executable, script, download_path, assembly,
                        '--metadata_path', metadata_file,
                        '--min_chip_per_cell',str(min_chip_per_cell),
                        '--min_cells_per_chip', str(min_cells_per_chip),
                        '--all_regions_file',all_regions_file_unfiltered
                        ]
logger.info(' '.join(parallel_cmd))
process = subprocess.Popen(parallel_cmd)
stdout = process.communicate()[0]
logger.info(stdout)
rc = process.returncode
if rc != 0:
    raise Exception("%s failed with error %i" % (' '.join(parallel_cmd), rc))

# create matrix or load in existing
matrix_path_all = os.path.join(download_path, 'train_total.h5') # all sites

# written in parralel_download.py
row_df = pd.read_csv(os.path.join(download_path, "row_df.csv"))

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
# remove h5 file with all zeros
os.remove(matrix_path_all) # remove h5 file with all zeros
