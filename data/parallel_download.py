# Code for running file downloads in parallel

# imports 
from multiprocessing import Pool
import multiprocessing as mp
import argparse
from download_functions import *
import os
import pandas as pd
import numpy as np
import pyranges as pr

logger = set_logger(__name__)

# number of threads
threads = mp.cpu_count()
logger.info("%i threads available for processing" % threads)

##############################################################################################
############################################# PARSE USER ARGUMENTS ###########################
##############################################################################################

# Parser for user specific locations
parser = argparse.ArgumentParser(description='Downloads bed files in parallel and processes them to npy files')

parser.add_argument('download_path', help='Temporary path to download temporary files to.', type=str)
parser.add_argument('assembly', help='assembly to filter files in metadata.tsv file by.', choices=['ce10', 'ce11', 'dm3', 'dm6', 'hg19', 'hg38', 'mm10', 'mm9', 'rn6', 'sacCer3'], type=str)

parser.add_argument('--metadata_path',type=str, 
                    help='Path to ChIP-Atlas metadata csv file.')

parser.add_argument('--min_chip_per_cell', help='Minimum ChIP-seq experiments for each cell type.', type=int, default=1)
parser.add_argument('--min_cells_per_chip', help='Minimum cells a given ChIP-seq target must be observed in.', type=int, default=3)

parser.add_argument('--all_regions_file', help='File to read regions from', type=str, default=None)
parser.add_argument('--bgzip', help='Path to bgzip executable', type=str, default='bgzip')

download_path = parser.parse_args().download_path
all_regions_file_unfiltered = parser.parse_args().all_regions_file_unfiltered
metadata_path = parser.parse_args().metadata_path
min_chip_per_cell = parser.parse_args().min_chip_per_cell
min_cells_per_chip = parser.parse_args().min_cells_per_chip

# TODO: need to define:
# repliate Groups
# nregions
# tmp_download_path, bed_download_path


# create matrix or load in existing
matrix_path_all = os.path.join(download_path, 'train_total.h5') # all sites

# collect all regions and merge by chromsome, count number of 200bp bins
pyDF = pr.read_bed(all_regions_file_unfiltered)

def processGroups(n):
    '''
    Process set of enumerated dataframe rows, a group of (antigen, cell types)

    Args:
        :param n: row from a grouped dataframe, ((antigen, celltype), samples)
        :param tmp_download_path: where npz files should be saved to

    :return tuple: tuple of (tmp_file_save, cell, target)

    '''
    target, cell = n[0] # tuple of  ((antigen, celltype), samples)
    samples = n[1]

    id_ = samples.iloc[0]['Experimental ID'] # just use first as ID for filename

    if target == 'DNase-Seq' or target == 'DNase-seq':
        target = target.split("-")[0] # remove "Seq/seq"

    # create a temporaryfile
    # save appends 'npy' to end of filename
    tmp_file_save = os.path.join(tmp_download_path, id_)

    # if there is data in this row, it was already written, so skip it.
    if os.path.exists(tmp_file_save + ".npz"):
        logger.info("Skipping %s, %s, already written to %s" % (target,cell, tmp_file_save))
        arr = np.load(tmp_file_save + ".npz", allow_pickle=True)['data'].astype('i1') # int8
    else:
        logger.info("writing into matrix for %s, %s" % (target, cell))

        downloaded_files = [download_url(sample, bed_download_path) for i, sample in samples.iterrows()]

        # filter out bed files with less than 200 peaks
        downloaded_files = list(filter(lambda x: count_lines(x) > 200, downloaded_files))

        arr = lojs_overlap(downloaded_files, pyDF)

        np.savez_compressed(tmp_file_save, data=arr)

    if np.sum(arr) == 0:
        return None
    else:
        return (tmp_file_save, cell, target)



with Pool(threads) as p:
    # list of tuples for each file, where tuple is (i, filename, featurename)
    results = p.map(processGroups, replicate_groups)

results = [i for i in results if i is not None]

# load in cells and targets into a dataframe
cellTypes = [i[1] for i in results]
targets = [i[2] for i in results]
row_df = pd.DataFrame({'cellType': cellTypes,'target': targets})

### save matrix
if os.path.exists(matrix_path_all):
    h5_file = h5py.File(matrix_path_all, "w")
    # make sure the dataset hasnt changed if you are appending
    assert(h5_file[0,:].shape[0] == nregions)
    assert(h5_file[:,0].shape[0] == len(results))

else:
    h5_file = h5py.File(matrix_path_all, "w")
    matrix = h5_file.create_dataset("data", (len(results), nregions), dtype='i1', # int8
        compression='gzip', compression_opts=9)

    for i, (f, cell, target) in enumerate(results):

        matrix[i,:] = np.load(f + ".npz", allow_pickle=True)['data'].astype('i1') # int8

        if i % 100 == 0:
            logger.info("Writing %i, feature %s..." % (i, feature_name))

    h5_file.close()

logger.info("Done saving sparse data")