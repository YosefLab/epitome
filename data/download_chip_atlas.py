    

# ## Download all data for hg38 ChIP-Atlas
#
# This script uses files.txt and ENCODE metadata to download DNAse for hg19 for specific cell types.
# Because ENCODE does not have hg19 data for ATAC-seq, we have to re-align it from scratch.



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
import gzip
import logging
import traceback
import glob

##################################### LOG INFO ################################
logger = logging.getLogger('DOWNLOAD ChIP-Atlas')
logger.setLevel(logging.DEBUG)

# create console handler with a low log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(ch)

# number of threads
threads = mp.cpu_count()
logger.info("%i threads available for processing" % threads)

########################### Functions ########################################
def lojs_overlap(feature_files, compare_pr):
        """
        Function to run left outer join in features to all_regions_file

        feature_file: list of paths to file to run intersection with all_regions_file
        :return arr: array same size as the number of genomic regions in all_regions_file
        """
        
        # TODO: you can use the s2n value to filter out. Then you can use a consensus strategy.
        
        if len(feature_files) == 0:
            logger.warn("WARN: lojs_overlap failed for all files %s with 0 lines" % ','.join(feature_files))
            return np.zeros(len(compare_pr))
        
        #### Number of files that must share a consensus ####
        if len(feature_files)<=2:
            n = 1 # if there are 1-2 files just include all
        elif len(feature_files) >=3 and len(feature_files) <= 7:
            n=2
        else:
            n = int(len(feature_files)/4) # in 25% of files

        # Very slow: concatenate all bed files and only take regions with n overlap
        group_pr = pr.concat([pr.read_bed(i).merge(slack=20) for i in feature_files])
        group_pr = group_pr.merge(slack=20, count=True).df
        group_pr = group_pr[group_pr['Count']>=n]

        # Remove count column and save to bed file
        group_pr.drop('Count', inplace=True, axis=1)

        pr1 = pr.PyRanges(group_pr)

        intersected = compare_pr.count_overlaps(pr1)
        arr = intersected.df['NumberOverlaps'].values
        arr[arr>0] = 1
        return arr

##############################################################################################
############################################# PARSE USER ARGUMENTS ###########################
##############################################################################################

# Parser for user specific locations
parser = argparse.ArgumentParser(description='Downloads ChIP-Atlas data from a chip_atlas_experiment_list.csv file.')

parser.add_argument('download_path', help='Temporary path to download bed/bigbed files to.', type=str)
parser.add_argument('assembly', help='assembly to filter files in metadata.tsv file by.', choices=['ce10', 'ce11', 'dm3', 'dm6', 'hg19', 'hg38', 'mm10', 'mm9', 'rn6', 'sacCer3'], type=str)
parser.add_argument('output_path', help='path to save file data to', type=str)

parser.add_argument('--metadata_url',type=str, default="ftp://ftp.biosciencedbc.jp/archive/chip-atlas/LATEST/chip_atlas_experiment_list.zip",
                    help='ENCODE metadata URL.')

parser.add_argument('--min_chip_per_cell', help='Minimum ChIP-seq experiments for each cell type.', type=int, default=1)
parser.add_argument('--min_cells_per_chip', help='Minimum cells a given ChIP-seq target must be observed in.', type=int, default=3)

parser.add_argument('--regions_file', help='File to read regions from', type=str, default=None)
parser.add_argument('--bgzip', help='Path to bgzip executable', type=str, default='bgzip')


download_path = parser.parse_args().download_path
assembly = parser.parse_args().assembly
output_path = parser.parse_args().output_path
metadata_path = parser.parse_args().metadata_url
min_chip_per_cell = parser.parse_args().min_chip_per_cell
min_cells_per_chip = parser.parse_args().min_cells_per_chip
all_regions_file_unfiltered = parser.parse_args().regions_file
bgzip = parser.parse_args().bgzip

# where to temporarily store np files
tmp_download_path = os.path.join(download_path, "tmp_np")

# where to cleanly store data downloaded from ChIP-Atlas
bed_download_path = os.path.join("/data/yosef/epitome/ChIP_atlas/", assembly, "downloads/bed/")
assert(len(glob.glob(os.path.join(bed_download_path, "*.05.bed"))) > 1000 )

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

# gzipped tmp file
all_regions_file_unfiltered_gz = all_regions_file_unfiltered + ".gz"

# download metadata if it does not exist
metadata_file = os.path.join(download_path, os.path.basename(metadata_path).replace('.zip','.csv'))

if not os.path.exists(metadata_file):
    zipped = os.path.join(download_path, os.path.basename(metadata_path))
    
    if not os.path.exists(zipped):
        subprocess.check_call(["wget", "-O", zipped, "-np", "-r", "-nd", metadata_path])
    
    # gunzip file
    subprocess.check_call(["unzip", zipped])

files = pd.read_csv(metadata_file, engine='python') # needed for decoding

##############################################################################################
######### get all files that are peak files for histone marks or TF ChiP-seq #################
##############################################################################################

# assembly column is either 'Assembly' or 'File assembly'
assembly_column = files.filter(regex=re.compile('Assembly', re.IGNORECASE)).columns[0]

antigen_classes = ['DNase-seq','Histone','TFs and others']

assembly_files = files[(files[assembly_column] == assembly)& 
                                   (files['Antigen class'].isin(antigen_classes))]

# Get unique by Antigen class, Antigen, Cell type class, Cell type, Cell type description. 
rm_dups = assembly_files[['Antigen class', 'Antigen', 'Cell type class', 'Cell type', 'Cell type description']].drop_duplicates()
filtered_files = assembly_files.loc[rm_dups.index]

# get unique dnase experiments
filtered_dnase = filtered_files[((filtered_files["Antigen class"] == "DNase-seq"))]

chip_files = filtered_files[(((filtered_files["Antigen class"] == 'Histone') | (filtered_files["Antigen class"] == 'TFs and others')))]

# only want ChIP-seq from cell lines that have DNase
filtered_chip = chip_files[(chip_files["Cell type"].isin(filtered_dnase["Cell type"]))]

# only want assays that are shared between more than 3 cells
filtered_chip = filtered_chip.groupby("Antigen").filter(lambda x: len(x) >= min_cells_per_chip)

# only want cells that have more than min_chip_per_cell epigenetic marks
filtered_chip = filtered_chip.groupby("Cell type").filter(lambda x: len(x) >= min_chip_per_cell)

# only filter if use requires at least one chip experiment for a cell type.
if min_chip_per_cell > 0:
    # only want DNase that has chip.
    filtered_dnase = filtered_dnase[(filtered_dnase["Cell type"].isin(filtered_chip["Cell type"]))]

# combine dataframes
filtered_files = filtered_dnase.append(filtered_chip)
filtered_files.reset_index(inplace = True)

# group by antigen/celltype combinations. Iterate over these
replicate_groups = assembly_files[(assembly_files['Antigen'].isin(filtered_files['Antigen'])) & 
                                  (assembly_files['Cell type'].isin(filtered_files['Cell type']))] 

# read in annotated Antigens
TF_categories = pd.read_csv('/home/eecs/akmorrow/EPITOME/TF_generalized_binding/data/ChIP_target_types.csv',sep=',')
TF_categories.replace({'DNase': 'DNase-Seq'}, inplace=True)

# sanity check that all antigens are accounted for in TF_categories
assert len([i for i in set(replicate_groups['Antigen']) if i not in list(TF_categories['Name'])]) == 0

# Filter out ChIP-seq not in TFs, accessibility, histones, etc. We lose about 1100 rows
filtered_names = TF_categories[TF_categories['Group'].isin(['TF','chromatin accessibility','chromatin modifier','histone',
 'histone modification'])]

replicate_groups = replicate_groups[replicate_groups['Antigen'].isin(filtered_names['Name'])]

################## Filter by signal to noise values ###########################
if assembly != "hg38":
    raise Exception("Error: this was hardcoded for hg38, you need to compute signal to noise values for assembly %s" % assembly)

# read in signal to noise 
tmp_path = os.path.join('/data/yosef2/scratch/users/akmorrow/epitome/chip_atlas/', "tmp")
def readSignalToNoise(x):
    f = x[1]
    # check if bed file exists. if not, download it
    id_ = f["Experimental ID"]

    # where to write s2n info
    tmp_output = os.path.join(tmp_path, "%s.txt" % id_)
    
    if os.path.exists(tmp_output):
        with open(tmp_output, "r") as f:
            data = f.read().split(",")
            ret = [id_]
            ret.extend([float(i) for i in data])
            return ret
    else:
        return [id_, np.NAN, np.NAN, np.NAN]


with mp.Pool(threads) as p:
    # list of tuples for each file, where tuple is (i, filename, featurename)
    results = p.map(readSignalToNoise, replicate_groups.iterrows())
    
# save final results
s2nData = pd.DataFrame(results)
s2nData.columns = ["ID", "signal", "noise", "background"]
s2nData['s2n'] = s2nData['signal']/s2nData['noise']

# filter samples with signal to noise > 0.05
VALID_IDs = s2nData[s2nData['s2n']>0.05]

replicate_groups = replicate_groups[replicate_groups['Experimental ID'].isin(VALID_IDs['ID'])]
replicate_groups.reset_index(inplace = True)

logger.info("Processing %i antigens and %i experiments" % (len(set(replicate_groups['Antigen'])), len(replicate_groups)))

# group experiments together
replicate_groups = replicate_groups.groupby(['Antigen', 'Cell type'])

##############################################################################################
##################################### download all files #####################################
##############################################################################################

def download_url(f, tries = 0):
    '''
    Downloads a file from filtered_files dataframe row
    
    Returns path to downloaded file
    '''

    path = f["Peak-call (BED) (q < 1E-05)"]
    id_ = f["Experimental ID"]
    file_basename = os.path.basename(path)

    if tries == 2:
        raise Exception("File accession %s from URL %s failed for download 3 times. Exiting 1..." % (id_, path))

    outname_bed = os.path.join(bed_download_path, file_basename)

    # make sure file does not exist before downloading
    try:
        if not os.path.exists(outname_bed):

            logger.warning("Trying to download %s for the %ith time..." % (path, tries))

            if sys.version_info[0] < 3:
                # python 2
                urllib.urlretrieve(path, filename=outname_bed)
            else:
                # python 3
                urllib.request.urlretrieve(path, filename=outname_bed)

    except:
        # increment tries by one and re-try download
        return download_url(f, tries + 1)
        
    return outname_bed

##############################################################################################
############################# window chromsizes into 200bp ###################################
##############################################################################################

# get chrom sizes file and make windows for genome
if not os.path.exists(all_regions_file_unfiltered):
    tmpFile = all_regions_file_unfiltered + ".tmp"
    chrom_sizes_file = os.path.join(download_path, "%s.chrom.sizes" % assembly)

    # download file
    if not os.path.exists(chrom_sizes_file):
        if assembly == "hg38":
            # special case => hg38 in UCSC
            subprocess.check_call(["wget", "-O", chrom_sizes_file, "-np", "-r", "-nd", "https://genome.ucsc.edu/goldenPath/help/%s.chrom.sizes" % 'hg38'])
        else:
            subprocess.check_call(["wget", "-O", chrom_sizes_file, "-np", "-r", "-nd", "https://genome.ucsc.edu/goldenPath/help/%s.chrom.sizes" % assembly])

    # window genome into 200bp regions
    if not os.path.exists(tmpFile):
        stdout = open(tmpFile,"wb")
        subprocess.call(["bedtools", "makewindows", "-g", chrom_sizes_file, "-w", "200"],stdout=stdout)
        stdout.close()

    # filter out chrM, _random and _cl chrs
    stdout = open(all_regions_file_unfiltered,"wb")
    subprocess.check_call(["grep", "-vE", "_|chrM|chrM|chrX|chrY", tmpFile], stdout = stdout)
    stdout.close()

# zip and index pos file
# used in inference for faster file reading.
if not os.path.exists(all_regions_file_unfiltered_gz):

    stdout = open(all_regions_file_unfiltered_gz,"wb")
    subprocess.call([bgzip, "--index", "-c", all_regions_file_unfiltered],stdout=stdout)
    stdout.close()


# get number of genomic regions in all.pos.bed file
nregions = sum(1 for line in open(all_regions_file_unfiltered))
logger.info("Completed windowing genome with %i regions" % nregions)

#############################################################################################
################################ save all files to matrix ###################################
#############################################################################################

# create matrix or load in existing
matrix_path_all = os.path.join(download_path, 'train_total.h5') # all sites
matrix_path = os.path.join(download_path, 'train.h5')           # filtered nonzero sites

# collect all regions and merge by chromsome, count number of 200bp bins
pyDF = pr.read_bed(all_regions_file_unfiltered)
                      
def processGroups(n):
    '''
    Process set of enumerated rows, a group of (antigen, cell types)
    
    '''
    iter_ =n[0] #int iteration
    target, cell = n[1][0] # tuple of  ((antigen, celltype), samples)
    samples = n[1][1]

    id_ = samples.iloc[0]['Experimental ID'] # just use first as ID for filename
    
    if target == 'DNase-Seq':
        target = target.split("-")[0] # remove "Seq"

    feature_name = "%i\t%s|%s|%s" % (iter_+1, cell, target, "None")
    
    # create a temporaryfile 
    # save appends 'npy' to end of filename
    tmp_file_save = os.path.join(tmp_download_path, id_)
    
    # if there is data in this row, it was already written, so skip it.
    if os.path.exists(tmp_file_save + ".npz"):
        logger.info("Skipping index %i, already written to %s" % (iter_, tmp_file_save))
    else:
        logger.info("writing into matrix at positions %i" % (iter_))

        downloaded_files = [download_url(sample) for i, sample in samples.iterrows()]
        
        arr = lojs_overlap(downloaded_files, pyDF)
            
        np.savez_compressed(tmp_file_save, data=arr)

    return (iter_, tmp_file_save, feature_name)                         
                      
with Pool(threads) as p:
    # list of tuples for each file, where tuple is (i, filename, featurename)
    results = p.map(processGroups, enumerate(replicate_groups))

### save feature_name file
# feature_name file to write metadata
feature_name_file = os.path.join(output_path,"feature_name")
# write feature names
feature_name_handle = open(feature_name_file, 'w')
start = "0\tID" # first col 1 based numbering, second col is ID (IMR-90|H4K8ac|None), last is feature or label type
feature_name_handle.write("%s\n" % start)

for i, f, feature_name in results:

    # append to file and flush
    feature_name_handle.write("%s\n" % feature_name)

feature_name_handle.close()
                      
### save matrix
if os.path.exists(matrix_path_all):
    h5_file = h5py.File(matrix_path_all, "a")
    matrix = h5_file['data']

    # make sure the dataset hasnt changed if you are appending
    assert(matrix[0,:].shape[0] == nregions)
    assert(matrix[:,0].shape[0] == len(filtered_files))

else:
    h5_file = h5py.File(matrix_path_all, "w")
    matrix = h5_file.create_dataset("data", (len(filtered_files), nregions), dtype='i1', # int8
        compression='gzip', compression_opts=9)

for i, f, feature_name in results: 
    
        matrix[i,:] = np.load(f + ".npz", allow_pickle=True)['data'].astype('i1') # int8

        if i % 100 == 0:
            logger.info("Writing %i, feature %s..." % (i, feature_name))

h5_file.close()
         
logger.info("Done saving data")

# can read matrix back in using:
# > import h5py
# > tmp = h5py.File(os.path.join(download, 'train.h5'), "r")
# > tmp['data']



######################################################################################
###################### LOAD DATA BACK IN AND SAVE AS NUMPY ###########################
######################################################################################

def save_epitome_numpy_data(download_dir, output_path):
    """
    Saves epitome labels as numpy arrays, filtering out training data for 0 vectors.

    Args:
        :param download_dir: Directory containing train.h5, all.pos.bed file and feature_name file.
        :param output_path: new output path. saves as numpy files.

    """
    # paths to save 0 reduced files to
    all_regions_file = os.path.join(output_path, "all.pos.bed")
    all_regions_file_gz = all_regions_file + ".gz"

    if not os.path.exists(all_regions_file) or not os.path.exists(matrix_path):

        if not os.path.exists(output_path):
            os.mkdir(output_path)
            logger.info("%s Created " % output_path)

        # load in all data into RAM: much faster for indexing
        h5_data = h5py.File(matrix_path_all, "r")['data']
        
        
#         ### HACK: previously matrix_path_allwas saved as f4 type, and didn't fit into memory
#         # here we set dtype to i1 to get more space
#         # creates a new Dataset instance that points to the same HDF5 identifier
#         d_new = h5py.Dataset(t.id)

#         # set the ._local.astype attribute to the desired output type
#         d_new._local.astype = np.dtype('i1')
#         h5_data = d_new[:,:]
#         ## end hack. you can remove once this is rerun, as you have fixed lines 353-374 which 
#         ## set the type to i1
        
        logger.info("loaded data..")

        # get non-zero indices
        nonzero_indices = np.where(np.sum(h5_data, axis=0) > 0)[0]

        ## get data from columns where sum > 0
        nonzero_data = h5_data[:,nonzero_indices]
        logger.info("number of new indices in %i" % nonzero_indices.shape[0])

        # filter and re-save all_regions_file
        logger.info("saving new regions file")
        pyDF = pd.read_csv(all_regions_file_unfiltered_gz, sep='\t', header=None)
        pyDF.loc[nonzero_indices].to_csv(all_regions_file, index=False, sep='\t', header=None)
        logger.info("done saving regions file")

        logger.info("saving new matrix")
        # resave h5 file without 0 columns
        h5_file = h5py.File(matrix_path, "w")
        matrix = h5_file.create_dataset("data", nonzero_data.shape, dtype='i',
            compression='gzip', compression_opts=9)
        matrix[:,:] = nonzero_data

        h5_file.close()
        logger.info("done saving matrix")



    # gzip filtered all_regions_file
    if not os.path.exists(all_regions_file_gz):
        stdout = open(all_regions_file_gz,"wb")
        subprocess.call([bgzip, "--index", "-c", all_regions_file],stdout=stdout)
        stdout.close()


    train_output_np = os.path.join(output_path, "train.npz")
    valid_output_np = os.path.join(output_path, "valid.npz")
    test_output_np = os.path.join(output_path, "test.npz")

    if not os.path.exists(test_output_np):
        h5_file = h5py.File(matrix_path, "r")
        h5_data = h5_file['data']

        # split nonzero_data into train, valid, test
        EPITOME_TRAIN_REGIONS, EPITOME_VALID_REGIONS, EPITOME_TEST_REGIONS = calculate_epitome_regions(all_regions_file_gz)

        TRAIN_RANGE = np.r_[EPITOME_TRAIN_REGIONS[0][0]:EPITOME_TRAIN_REGIONS[0][1],
                        EPITOME_TRAIN_REGIONS[1][0]:EPITOME_TRAIN_REGIONS[1][1]]
        train_data = h5_data[:,TRAIN_RANGE]
        logger.info("loaded train..")

        valid_data = h5_data[:,EPITOME_VALID_REGIONS[0]:EPITOME_VALID_REGIONS[1]]
        logger.info("loaded valid..")

        test_data = h5_data[:,EPITOME_TEST_REGIONS[0]:EPITOME_TEST_REGIONS[1]]
        logger.info("loaded test..")

        # save files
        logger.info("saving sparse train.npz, valid.npz and test.npyz to %s" % output_path)

        scipy.sparse.save_npz(train_output_np, scipy.sparse.csc_matrix(train_data,dtype=np.int8))
        scipy.sparse.save_npz(valid_output_np, scipy.sparse.csc_matrix(valid_data, dtype=np.int8))
        scipy.sparse.save_npz(test_output_np, scipy.sparse.csc_matrix(test_data, dtype=np.int8))

        # To load back in sparse matrices, use:
        # > sparse_matrix = scipy.sparse.load_npz(train_output_np)
        # convert whole matrix:
        # > sparse_matrix.todense()
        # index in:
        # > sparse_matrix[:,0:10].todense()

        h5_file.close()



# finally, save outputs
save_epitome_numpy_data(download_path, output_path)

# rm tmp unfiltered bed files
os.remove(all_regions_file_unfiltered)
os.remove(all_regions_file_unfiltered_gz)
os.remove(all_regions_file_unfiltered + ".tmp")
# remove h5 file with all zeros
os.remove(matrix_path_all) # remove h5 file with all zeros
