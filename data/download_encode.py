

# ## Download DNase from ENCODE
#
# This script uses files.txt and ENCODE metadata to download DNAse for hg19 for specific cell types.
# Because ENCODE does not have hg19 data for ATAC-seq, we have to re-align it from scratch.



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
from itertools import islice
import scipy.sparse
from epitome.functions import *
import sys
import shutil
import gzip
import logging

##################################### LOG INFO ################################
logger = logging.getLogger('DOWNLOAD ENCODE')
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
threads = multiprocessing.cpu_count()
logger.info("%i threads available for processing" % threads)

########################### Functions ########################################
def chunk(it, size):
    """ for batching an iterator

    :param it: iterator
    :param size: size to batch

    :return batched iterator
    """
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def loj_overlap(feature_file):
        """
        Callback function to run left outer join in features to all_regions_file

        feature_file: path to file to run intersection with all_regions_file
        :return arr: array same size as the number of genomic regions in all_regions_file
        """
        # -c :For each entry in A, report the number of hits in B
        # -f: percent overlap in A
        # -loj: left outer join
        cmd = ['bedtools', 'intersect', '-c', '-a',
               all_regions_file_unfiltered, '-b',
               feature_file,
               '-f', '0.5', '-loj'] # 100 bp overlap (0.5 * 100)

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        out, err = process.communicate()
        out = out.decode('UTF-8').rstrip().split("\n")

        # array of 0/1s. 1 if epigenetic mark has an overlap, 0 if no overlap (. means no overlapping data)
        arr = np.array(list(map(lambda x: 0 if x.split("\t")[3] == '0' else 1, out)))
        return arr


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

parser.add_argument('--min_chip_per_cell', help='Minimum ChIP-seq experiments for each cell type.', type=int, default=10)
parser.add_argument('--min_cells_per_chip', help='Minimum cells a given ChIP-seq target must be observed in.', type=int, default=3)

parser.add_argument('--regions_file', help='File to read regions from', type=str, default=None)
parser.add_argument('--bgzip', help='Path to bgzip executable', type=str, default='bgzip')
parser.add_argument('--bigBedToBed', help='Path to bigBedToBed executable, downloaded from http://hgdownload.cse.ucsc.edu/admin/exe/', type=str, default='bigBedToBed')


download_path = parser.parse_args().download_path
assembly = parser.parse_args().assembly
bigBedToBed = parser.parse_args().bigBedToBed
output_path = parser.parse_args().output_path
metadata_path = parser.parse_args().metadata_url
min_chip_per_cell = parser.parse_args().min_chip_per_cell
min_cells_per_chip = parser.parse_args().min_cells_per_chip
all_regions_file_unfiltered = parser.parse_args().regions_file
bgzip = parser.parse_args().bgzip



# make paths if they do not exist
if not os.path.exists(download_path):
    os.makedirs(download_path)
if not os.path.exists(output_path):
    os.makedirs(output_path)

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
metadata_file = os.path.join(download_path, 'metadata.tsv')

if not os.path.exists(metadata_file):
    subprocess.check_call(["wget", "-O", metadata_file, "-np", "-r", "-nd", metadata_path])

files = pd.read_csv(metadata_file, sep="\t")

##############################################################################################
######### get all files that are peak files for histone marks or TF ChiP-seq #################
##############################################################################################

# assembly column is either 'Assembly' or 'File assembly'
assembly_column = files.filter(regex=re.compile('Assembly', re.IGNORECASE)).columns[0]

filtered_files = files[(files[assembly_column] == assembly) &
#                   (files["Biosample genetic modifications targets"].isnull()) & Need for ENCODE3
                  (files["Audit ERROR"].isnull()) &
                  (files["Biosample treatments"].isnull())]


# get unique dnase experiments
dnase_files = filtered_files[((filtered_files["Output type"] == "peaks") & (filtered_files["Assay"] == "DNase-seq"))]
# chose DNase files without errors, if possible
dnase_files = dnase_files.sort_values(by=['Audit WARNING','Audit NOT_COMPLIANT'])
filtered_dnase = dnase_files.drop_duplicates(subset=["Biosample term name"] , keep='last')

chip_files = filtered_files[(((filtered_files["Output type"] == "replicated peaks") | (filtered_files["Output type"] == "optimal IDR thresholded peaks"))
                             & (filtered_files["Assay"].str.contains("ChIP-seq")) )] # or conservative idr thresholded peaks?

# only want ChIP-seq from cell lines that have DNase
filtered_chip = chip_files[(chip_files["Biosample term name"].isin(filtered_dnase["Biosample term name"]))]
# select first assay without audit warning
filtered_chip = filtered_chip.sort_values(by=['Audit WARNING','Audit NOT_COMPLIANT'])
filtered_chip = filtered_chip.drop_duplicates(subset=["Biosample term name","Experiment target"] , keep='last')




# only want assays that are shared between more than 3 cells
filtered_chip = filtered_chip.groupby("Experiment target").filter(lambda x: len(x) >= min_cells_per_chip)

# only want cells that have more than min_chip_per_cell epigenetic marks
filtered_chip = filtered_chip.groupby("Biosample term name").filter(lambda x: len(x) >= min_chip_per_cell)

# only filter if use requires at least one chip experiment for a cell type.
if min_chip_per_cell > 0:
    # only want DNase that has chip.
    filtered_dnase = filtered_dnase[(filtered_dnase["Biosample term name"].isin(filtered_chip["Biosample term name"]))]

# get ATAC-seq data
filtered_atac = filtered_files[(filtered_files["Assay"].str.contains("ATAC-seq")) &  (filtered_files["Output type"] == "IDR thresholded peaks")] 
filtered_atac = filtered_atac.drop_duplicates(subset=["Biosample term name"] , keep='last')
filtered_atac = filtered_atac[(filtered_atac["Biosample term name"].isin(filtered_dnase["Biosample term name"]))]

# combine dataframes
filtered_files = filtered_dnase.append(filtered_chip).append(filtered_atac)
logger.info("Processing %i files..." % len(filtered_files))

##############################################################################################
##################################### download all files #####################################
##############################################################################################

def download_url(f, tries = 0):

    logger.warning("Trying to download %s for the %ith time..." % (f["File download URL"], tries))

    if tries == 2:
        raise Exception("File accession %s from URL %s failed for download 3 times. Exiting 1..." % (f['File accession'], f["File download URL"]))

    path = f["File download URL"]
    ext = path.split(".")[-1]
    if (ext == "gz" and path.split(".")[-2]  == 'bed'):
        ext = "bed.gz"

    id = f["File accession"]
    # file name == accession__target-species__cellline.bigbed
    target = f["Assay"] if str(f["Experiment target"]) == "nan" else f["Experiment target"]
    if (target == "DNase-seq"):
        target = "DNase" # strip for consistency

    base_file = os.path.join(download_path, "%s_%s_%s" % (id, target, f["Biosample term name"].replace('/','-'))) # issue with filename
    base_file = base_file.replace(' ','-') # remove spaces
    base_file = base_file.replace(',','') # remove commas


    outname_bb = "%s.%s" % (base_file, ext)
    outname_bed = "%s.%s" % (base_file, "bed")

    # make sure file does not exist before downloading
    try:
        if not os.path.exists(outname_bed):

            # download if not yet downloaded
            if not os.path.exists(outname_bb):
                if sys.version_info[0] < 3:
                    # python 2
                    urllib.urlretrieve(path, filename=outname_bb)
                else:
                    # python 3
                    urllib.request.urlretrieve(path, filename=outname_bb)

            if (ext == "bed.gz"):
                subprocess.check_call(["gunzip","-f",outname_bb])
            elif (ext == "bigBed"):
                subprocess.check_call([bigBedToBed, outname_bb, outname_bed])
                os.remove(outname_bb)
    except:
        # increment tries by one and re-try download
        download_url(f, tries + 1)

# download all files
rows = list(map(lambda x: x[1], filtered_files.iterrows()))
pool = multiprocessing.Pool(processes=threads)
pool.map(download_url, rows)
pool.close()

##############################################################################################
############################# window chromsizes into 200bp ###################################
##############################################################################################

# get chrom sizes file and make windows for genome
if not os.path.exists(all_regions_file_unfiltered):
    tmpFile = all_regions_file_unfiltered + ".tmp"
    chrom_sizes_file = os.path.join(download_path, "%s.chrom.sizes" % assembly)

    # download file
    if not os.path.exists(chrom_sizes_file):
        if assembly == "GRCh38":
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

# read in already written features if they exist
feature_name_file = os.path.join(output_path,"feature_name")
if os.path.exists(feature_name_file):
    with open(feature_name_file) as f:
        written_features = f.readlines()
    written_features = [x.strip() for x in written_features]
else:
    written_features = []

# open feature file and write first row
feature_name_handle = open(feature_name_file, 'a+')
start = "0\tAuto-select"
if (start not in written_features):
    feature_name_handle.write("%s\n" % start)

# create matrix or load in existing
matrix_path_all = os.path.join(download_path, 'train_total.h5') # all sites
matrix_path = os.path.join(download_path, 'train.h5')           # filtered nonzero sites

if os.path.exists(matrix_path_all):
    h5_file = h5py.File(matrix_path_all, "a")
    matrix = h5_file['data']

    # make sure the dataset hasnt changed if you are appending
    assert(matrix[0,:].shape[0] == nregions)
    assert(matrix[:,0].shape[0] == len(filtered_files))

else:
    h5_file = h5py.File(matrix_path_all, "w")
    matrix = h5_file.create_dataset("data", (len(filtered_files), nregions), dtype='i8',
        compression='gzip', compression_opts=9)



bed_files = list(filter(lambda x: x.endswith(".bed") & x.startswith("ENC"), os.listdir(download_path)))
logger.info("Running bedtools on %i files..." % len(bed_files))

# batch files and parallelize
for b in chunk(enumerate(bed_files), threads):

    files = list(map(lambda x: os.path.join(download_path, x[1]), b))
    indices = [i[0] for i in b]
    cells = [fileName.split("_")[-1].split(".")[0] for (idx, fileName) in b] # remove file ext
    targets = [fileName.split("_")[1].split("-")[0] for (idx, fileName) in b] # remove "human"
    feature_names = ["%i\t%s|%s|%s" % (i+1, cell, target, "None") for (i, cell, target) in zip(indices, cells, targets)]

    # if whole batch is already written, skip it
    if (len(written_features) > indices[-1]+1):
        if (feature_names == written_features[indices[0]+1:indices[-1]+2]):
            logger.info("skipping batch for indices %s, already written" % indices)
            continue
        else:
            raise Exception("Feature name %s starting at position %i did not match feature file (%s). This is most likely because you \
            downloaded more or deleted bed files. Delete your saved in %s data files and start from scratch." \
            % (feature_names, indices[0], written_features[indices[0]+1:indices[-1]+1], download_path))


    logger.info("writing into matrix at positions %i:%i" % (indices[0], indices[-1]+1))

    # Should not parallelize bedtools. Gives non-deterministic results.
    for j, file in enumerate(files):
        matrix[indices[j],:] = loj_overlap(file)
        logger.info("writing file %s to index %i. Sum = %i" % (file, indices[j], np.sum(matrix[indices[j],:])))

    for feature_name in feature_names:

        logger.info("Writing metadata for %s" % (feature_name))

        # append to file and flush
        feature_name_handle.write("%s\n" % feature_name)

    feature_name_handle.flush()
    h5_file.flush()

feature_name_handle.close()
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
        h5_data = h5py.File(matrix_path_all, "r")['data'][:,:]
        logger.info("loaded data..")

        # get non-zero indices
        nonzero_indices = np.where(np.sum(h5_data, axis=0) > 0)[0]

        ## get data from columns where sum > 0
        nonzero_data = h5_data[:,nonzero_indices]
        logger.info("number of new indices in %i" % nonzero_indices.shape[0])

        # filter and re-save all_regions_file
        logger.info("saving new regions file")
        with gzip.open(all_regions_file_unfiltered_gz, 'rt') as f:
            # list of idx, chromosome
            lines = filter(lambda x: x[0] in nonzero_indices, enumerate(f.readlines()))
            mapped = list(map(lambda x: x[1], lines))
            with open(all_regions_file, "w") as new_f:
                new_f.writelines(mapped)
        logger.info("done saving regions file")

        logger.info("saving new matrix")
        # resave h5 file without 0 columns
        h5_file = h5py.File(matrix_path, "w")
        matrix = h5_file.create_dataset("data", nonzero_data.shape, dtype='i8',
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
