

# ## Download DNase from ENCODE
# 
# This script uses files.txt and ENCODE metadata to download DNAse for hg38 for specific cell types.
# Because ENCODE does not have hg19 data for ATAC-seq, we have to re-align it from scratch.



############################## Imports ####################################

import pandas as pd
import numpy as np
import os
import urllib
import multiprocessing
import subprocess
import math
import argparse
import h5py
from itertools import islice
import scipy.sparse
from epitome.functions import *

# number of threads
threads = multiprocessing.cpu_count()
print("%i threads available for processing" % threads)

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
               all_regions_file, '-b', 
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
parser.add_argument('bigBedToBed', help='Path to bigBedToBed executable, downloaded from http://hgdownload.cse.ucsc.edu/admin/exe/', type=str) 
parser.add_argument('output_path', help='path to save file data to', type=str)

download_path = parser.parse_args().download_path
assembly = parser.parse_args().assembly
bigBedToBed = parser.parse_args().bigBedToBed
output_path = parser.parse_args().output_path

# path to save regions to. must be defined before loj_overlap function
all_regions_file = os.path.join(output_path,"all.pos.bed")


# download metadata if it does not exist for this assembly
metadata_path = "http://v88x0-test-master.instance.encodedcc.org/metadata/type%%3DExperiment%%26assay_title%%3DTF%%2BChIP-seq%%26assay_title%%3DHistone%%2BChIP-seq%%26assay_title%%3DDNase-seq%%26assay_title%%3DATAC-seq%%26assembly%%3D%s%%26files.file_type%%3DbigBed%%2BnarrowPeak/metadata.tsv" % assembly


# download metadata for this assembly if it does not exist
metadata_file = os.path.join(download_path, 'metadata_%s.tsv' % assembly)

if not os.path.exists(metadata_file):
    subprocess.check_call(["wget", "-O", metadata_file, "-np", "-r", "-nd", metadata_path])

files = pd.read_csv(metadata_file, sep="\t")

##############################################################################################
######### get all files that are peak files for histone marks or TF ChiP-seq #################
##############################################################################################

filtered_files = files[(files["Biosample type"] == "cell line") & 
                  (files["Assembly"] == assembly) & 
                  (files["Biosample genetic modifications targets"].isnull())] # or conservative idr thresholded peaks?

# TODO: make sure no treatment!
# get unique dnase experiements
dnase_files = filtered_files[((filtered_files["Output type"] == "peaks") & (filtered_files["Assay"] == "DNase-seq"))]
dnase_files.drop_duplicates(subset=["Biosample term name"] , keep='first', inplace=True)

chip_files = filtered_files[((filtered_files["Output type"] == "optimal idr thresholded peaks") & (filtered_files["Assay"] == "ChIP-seq"))] # or conservative idr thresholded peaks?


# only want ChIP-seq from cell lines that have DNase
filtered_chip = chip_files[(chip_files["Biosample term name"].isin(dnase_files["Biosample term name"]))]

# only want cells that have more than 10 epigenetic marks
filtered_chip = filtered_chip.groupby("Biosample term name").filter(lambda x: len(x) > 10)

# only want assays that are shared between more than 2 cells
filtered_chip = filtered_chip.groupby("Experiment target").filter(lambda x: len(x) > 2)

# only want DNase from cell lines that have Chip-seq
filtered_dnase = dnase_files[(dnase_files["Biosample term name"].isin(filtered_chip["Biosample term name"]))]

# combine dataframes
filtered_files = filtered_dnase.append(filtered_chip)
print("Processing %i files..." % len(filtered_files))


##############################################################################################
##################################### download all files #####################################
##############################################################################################

def download_url(f):
    path = f["File download URL"]
    ext = path.split(".")[-1]
    id = f["File accession"]
    # file name == accession__target-species__cellline.bigbed
    target = f["Assay"] if str(f["Experiment target"]) == "nan" else f["Experiment target"]
    if (target == "DNase-seq"):
        target = "DNase" # strip for consistency
        
    base_file = os.path.join(download_path, "%s_%s_%s" % (id, target, f["Biosample term name"]))
    outname_bb = "%s.%s" % (base_file, ext)
    outname_bed = "%s.%s" % (base_file, "bed")
    
    # make sure file does not exist before downloading
    if not os.path.exists(outname_bed):
        # TODO python 3
        # urllib.request.urlretrieve(path, filename=outname_bb)
        # python 2
        urllib.urlretrieve(path, filename=outname_bb)
        subprocess.check_call([bigBedToBed, outname_bb, outname_bed])
    
# download all files
rows = list(map(lambda x: x[1], filtered_files.iterrows()))
pool = multiprocessing.Pool(processes=threads)
pool.map(download_url, rows)

print("Completed download")

##############################################################################################
############################# window chromsizes into 200bp ###################################
##############################################################################################

# get chrom sizes file and make windows for genome
if not os.path.exists(all_regions_file):
    tmpFile = all_regions_file + ".tmp"
    chrom_sizes_file = os.path.join(download_path, "%s.chrom.sizes" % assembly)
    # download file
    if not os.path.exists(chrom_sizes_file):
        subprocess.check_call(["wget", "-O", chrom_sizes_file, "-np", "-r", "-nd", "https://genome.ucsc.edu/goldenPath/help/%s.chrom.sizes" % assembly])
        
    # window genome into 200bp regions
    if not os.path.exists(tmpFile):
        stdout = open(tmpFile,"wb")
        subprocess.call(["bedtools", "makewindows", "-g", chrom_sizes_file, "-w", "200"],stdout=stdout)
        stdout.close()
    
    # filter out chrM, _random and _cl chrs
    stdout = open(all_regions_file,"wb")
    subprocess.check_call(["grep", "-vE", "_|chrM|chrM|chrX|chrY", tmpFile], stdout = stdout)
    stdout.close()
    
# zip and index pos file
# used in inference for faster file reading.
if not os.path.exists(all_regions_file + ".gz"):
    
    stdout = open(all_regions_file + ".gz","wb")
    subprocess.call(["bgzip", "--index", "-c", all_regions_file],stdout=stdout)
    stdout.close()


# get number of genomic regions in all.pos.bed file
nregions = sum(1 for line in open(all_regions_file)) 
print("Completed windowing genome with %i regions" % nregions)

##############################################################################################
################################# save all files to matrix ###################################
##############################################################################################

# read in already written features
feature_name_file = os.path.join(output_path,"feature_name")
with open(feature_name_file) as f:
    written_features = f.readlines()
written_features = [x.strip() for x in written_features] 

# open feature file and write first row
feature_name_handle = open(feature_name_file, 'a+')
start = "0\tAuto-select"
if (start not in written_features):
    feature_name_handle.write("%s\n" % start)

# create matrix or load in existing
matrix_path = os.path.join(download_path, 'train.h5')
if os.path.exists(matrix_path):
    h5_file = h5py.File(matrix_path, "a")
    matrix = h5_file['data']
    
    # make sure the dataset hasnt changed if you are appending
    assert(matrix[0,:].shape[0] == nregions)
    assert(matrix[:,0].shape[0] == len(filtered_files))
    
else:
    h5_file = h5py.File(matrix_path, "w")
    matrix = h5_file.create_dataset("data", (len(filtered_files), nregions), dtype='i',
        compression='gzip', compression_opts=9)

                 
bed_files = list(filter(lambda x: x.endswith(".bed") & x.startswith("ENC"), os.listdir(download_path)))
print("Running bedtools on %i files..." % len(bed_files))

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
            print("skipping batch for indices %s, already written" % indices)
            continue
        else:
            raise Exception("Feature name %s starting at position %i did not match feature file (%s). This is most likely because you \
            downloaded more or deleted bed files. Delete your saved in %s data files and start from scratch." \
            % (feature_names, indices[0], written_features[indices[0]+1:indices[-1]+1], download_path))
            

    print("writing into matrix at positions %i:%i" % (indices[0], indices[-1]+1))
    matrix[indices[0]:indices[-1]+1,:] = np.reshape(pool.map(loj_overlap, files), [len(files),-1])
    
    for feature_name in feature_names:
        
        print("Writing metadata for %s" % (feature_name))
        
        # append to file and flush
        feature_name_handle.write("%s\n" % feature_name)
    
    feature_name_handle.flush()
    h5_file.flush()

feature_name_handle.close()
h5_file.close()
print("Done saving data")

# can read matrix back in using:
# > import h5py
# > tmp = h5py.File(os.path.join(download, 'train.h5'), "r")
# > tmp['data']

######################################################################################
###################### LOAD DATA BACK IN AND SAVE AS NUMPY ###########################
######################################################################################




def save_epitome_numpy_data(data_dir, output_path):
    """
    Saves epitome labels as numpy arrays, filtering out training data for 0 vectors.
    
    Args:
        :param data_dir: Directory containing train.h5, all.pos.bed file and feature_name file.
        :param output_path: new output path. saves as numpy files.

    """
    # paths to save 0 reduced files to
    name = all_regions_file.split("/")[-1]
    new_regions_file = os.path.join(output_path, name)
    matrix_path = os.path.join(output_path, 'train.h5')
    
    if not os.path.exists(new_regions_file) and not os.path.exists(matrix_path):
        
        if not os.path.exists(output_path):
            os.mkdir(output_path)
            print("%s Created " % output_path)

        h5_path = os.path.join(data_dir, "train.h5")
        h5_data = h5py.File(h5_path, "r")['data']
        print("loaded data..")

        # get chunks for parallel processing
        chunks = list(chunk(range(h5_data.shape[1]), 1000000))

        def get_nonzero_indices(indices):
            start = indices[0]
            print(start)
            data = h5_data[:,indices[0]:indices[-1]]
            idx = np.where(np.sum(data, axis=0) > 0)[0]
            return data[:, idx], idx + start

        # get non-zero indices
        nonzero_ret = list(map(lambda x: get_nonzero_indices(x), chunks))

        ## combine indices and data
        nonzero_data = np.hstack(list(map(lambda x: x[0],  nonzero_ret)))
        nonzero_indices = np.hstack(list(map(lambda x: x[1],  nonzero_ret)))
        print("number of new indices in %i" % nonzero_indices.shape[0])

        # filter and re-save all_regions_file
        print("saving new file")
        with open(all_regions_file) as f:
            # list of idx, chromosome
            lines = filter(lambda x: x[0] in nonzero_indices, enumerate(f.readlines()))
            mapped = list(map(lambda x: x[1], lines))
            with open(new_regions_file, "w") as new_f:
                new_f.writelines(mapped)
        print("done saving file")

        print("saving new matrix")
        # resave h5 file without 0 columns
        h5_file = h5py.File(matrix_path, "w")
        matrix = h5_file.create_dataset("data", nonzero_data.shape, dtype='i',
        compression='gzip', compression_opts=9)
        matrix[:,:] = nonzero_data
        print("done saving matrix")
        
        h5_data.close()
        
        
    train_output_np = os.path.join(output_path, "train.npz")
    valid_output_np = os.path.join(output_path, "valid.npz")
    test_output_np = os.path.join(output_path, "test.npz")
    
    if not os.path.exists(test_output_np):
        
        h5_file = h5py.File(matrix_path, "r")
        h5_data = h5_file['data']
    
        # split nonzero_data into train, valid, test
        EPITOME_TRAIN_REGIONS, EPITOME_VALID_REGIONS, EPITOME_TEST_REGIONS = calculate_epitome_regions(new_regions_file)

        TRAIN_RANGE = np.r_[EPITOME_TRAIN_REGIONS[0][0]:EPITOME_TRAIN_REGIONS[0][1],
                        EPITOME_TRAIN_REGIONS[1][0]:EPITOME_TRAIN_REGIONS[1][1]]
        train_data = h5_data[:,TRAIN_RANGE]
        print("loaded train..")

        valid_data = h5_data[:,EPITOME_VALID_REGIONS[0]:EPITOME_VALID_REGIONS[1]]
        print("loaded valid..")

        test_data = h5_data[:,EPITOME_TEST_REGIONS[0]:EPITOME_TEST_REGIONS[1]]
        print("loaded test..")

        # save files
        print("saving sparse train.npz, valid.npz and test.npyz to %s" % output_path)
    
        scipy.sparse.save_npz(train_output_np, scipy.sparse.csc_matrix(train_data))
        scipy.sparse.save_npz(valid_output_np, scipy.sparse.csc_matrix(valid_data))
        scipy.sparse.save_npz(test_output_np, scipy.sparse.csc_matrix(test_data))
        
        # To load back in sparse matrices, use:
        # > sparse_matrix = scipy.sparse.load_npz(train_output_np)
        # convert whole matrix:
        # > sparse_matrix.todense()
        # index in:
        # > sparse_matrix[:,0:10].todense()
        
        h5_file.close()


    
# finally, save outputs
save_epitome_numpy_data(download_path, output_path)
