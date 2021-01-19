

# ## Download DNase from ENCODE
#
# This script uses files.txt and ENCODE metadata to download DNAse for hg19 for specific cell types.
# Because ENCODE does not have hg19 data for ATAC-seq, we have to re-align it from scratch.

# TODO: update parallel_download.py for ENCODE

############################## Imports ####################################

import pandas as pd
import numpy as np
import os
import urllib
import multiprocessing
from multiprocessing import Pool
import subprocess
import math
import argparse
import h5py
import re
from epitome.functions import *
import sys
import shutil
import logging

from download_functions import *

##################################### LOG INFO ################################
logger = set_logger()

########################### Functions ########################################
# TODO: delte, now using CHIPAtlas loj_overlap (its faster)
# def loj_overlap(feature_file):
#         """
#         Callback function to run left outer join in features to all_regions_file
#
#         feature_file: path to file to run intersection with all_regions_file
#         :return arr: array same size as the number of genomic regions in all_regions_file
#         """
#         # -c :For each entry in A, report the number of hits in B
#         # -f: percent overlap in A
#         # -loj: left outer join
#         cmd = ['bedtools', 'intersect', '-c', '-a',
#                all_regions_file_unfiltered, '-b',
#                feature_file,
#                '-f', '0.5', '-loj'] # 100 bp overlap (0.5 * 100)
#
#         process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
#         out, err = process.communicate()
#         out = out.decode('UTF-8').rstrip().split("\n")
#
#         # array of 0/1s. 1 if epigenetic mark has an overlap, 0 if no overlap (. means no overlapping data)
#         arr = np.array(list(map(lambda x: 0 if x.split("\t")[3] == '0' else 1, out)))
#         return arr


##############################################################################################
############################################# PARSE USER ARGUMENTS ###########################
##############################################################################################

# Parser for user specific locations
parser = argparse.ArgumentParser(description='Downloads ENCODE data from a metadata.tsv file from ENCODE.')

parser.add_argument('download_path', help='Temporary path to download bed/bigbed files to.', type=str)
parser.add_argument('assembly', help='assembly to filter files in metadata.tsv file by.', choices=['hg19','mm10','GRCh38'], type=str)
parser.add_argument('output_path', help='path to save file data to', type=str)

parser.add_argument('--metadata_url',type=str, default="http://www.encodeproject.org/metadata/type%3DExperiment%26assay_title%3DTF%2BChIP-seq%26assay_title%3DHistone%2BChIP-seq%26assay_title%3DDNase-seq%26assay_title%3DATAC-seq%26assembly%3Dhg19%26files.file_type%3DbigBed%2BnarrowPeak/metadata.tsv",
                    help='ENCODE metadata URL.')

parser.add_argument('--min_chip_per_cell', help='Minimum ChIP-seq experiments for each cell type.', type=int, default=1)
parser.add_argument('--min_cells_per_chip', help='Minimum cells a given ChIP-seq target must be observed in.', type=int, default=3)
parser.add_argument('--bigBedToBed', help='Path to bigBedToBed executable, downloaded from http://hgdownload.cse.ucsc.edu/admin/exe/', type=str, default='bigBedToBed')


download_path = parser.parse_args().download_path
assembly = parser.parse_args().assembly
output_path = parser.parse_args().output_path
metadata_path = parser.parse_args().metadata_url
min_chip_per_cell = parser.parse_args().min_chip_per_cell
min_cells_per_chip = parser.parse_args().min_cells_per_chip
bigBedToBed = parser.parse_args().bigBedToBed

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
metadata_file = os.path.join(meta_download_path, 'metadata_encode.tsv')

if not os.path.exists(metadata_file):
    subprocess.check_call(["wget", "-O", metadata_file, "-np", "-r", "-nd", metadata_path])

##############################################################################################
############################# window chromsizes into 200bp ###################################
##############################################################################################
window_genome(all_regions_file_unfiltered,
                download_path,
                assembly)

##############################################################################################
##################################### download all files #####################################
##############################################################################################

# call parallel download code
this_dir = os.path.dirname(os.path.abspath(__file__))
script = os.path.join(this_dir, 'parallel_download.py')
parallel_cmd = [sys.executable, script, download_path, assembly,
                        '--metadata_path', metadata_file,
                        '--min_chip_per_cell',str(min_chip_per_cell),
                        '--min_cells_per_chip', str(min_cells_per_chip),
                        '--all_regions_file',all_regions_file_unfiltered,
                        '--bigBedToBed', bigBedToBed
                        ]
logger.info(' '.join(parallel_cmd))
process = subprocess.Popen(parallel_cmd)
stdout = process.communicate()[0]
logger.info(stdout)
rc = process.returncode
if rc != 0:
    raise Exception("%s failed with error %i" % (' '.join(parallel_cmd), rc))

# path to matrix
matrix_path_all = os.path.join(download_path, 'train_total.h5') # all sites

# written in parralel_download.py. Contains cell/target information.
row_df_file = os.path.join(download_path, "row_df.csv")
row_df = pd.read_csv(row_df_file)

logger.info("Done saving sparse data")

# finally, save outputs
save_epitome_dataset(download_path,
                        output_path,
                        matrix_path_all,
                        all_regions_file_unfiltered,
                        row_df,
                        assembly,
                        "ENCODE")

# rm temporary files
os.remove(all_regions_file_unfiltered)
os.remove(row_df_file)
# remove h5 file with all zeros
os.remove(matrix_path_all)
