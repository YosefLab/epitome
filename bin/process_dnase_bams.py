# Process DNase raw bams to vectors

## Configurable filepaths

# path to deepsea .mat files
deepsea_path = "/data/akmorrow/epitome_data/deepsea_train/"

# deepsea positions file
_DEEPSEA_GENOME_REGIONS_FILENAME = "/home/eecs/akmorrow/epitome/data/allTFs.pos.bed"

# path to save processed data to
_PROCESSED_DATA_PATH = "/data/akmorrow/epitome_data/processed_dnase/"

# path to where dnase bams are stored. Bams need to be sorted and indexed. See bin/download_dnase_encode.sh for
# data processing
_ENCODE_DATA_PREFIX =  "/data/akmorrow/encode_data/"

# ## Imports

import pyDNase
from pyDNase import GenomicInterval
import h5py
import os
import linecache
import numpy as np
from scipy.io import loadmat
import scipy.sparse as ss
import h5sparse


# ## Define Constants

# DNase filepath dictionary

# DNase filepath dictionary
dnase_file_dict = {
  "A549": _ENCODE_DATA_PREFIX + "ENCFF414MBW_sorted_A549.bam",
  "HepG2": _ENCODE_DATA_PREFIX + "ENCFF224FMI_sorted_HepG2.bam",
  "K562": _ENCODE_DATA_PREFIX + "ENCFF678VYF_sorted_K562.bam",
  'GM12878':_ENCODE_DATA_PREFIX + "ENCFF775ZJX_sorted_GM12878.bam",
  'H1-hESC':_ENCODE_DATA_PREFIX + "ENCFF571SSA_sorted_H1heSC.bam",
  'HeLa-S3':_ENCODE_DATA_PREFIX + "ENCFF783TMX_sorted_HeLaS3.bam",
  'HUVEC':_ENCODE_DATA_PREFIX + "ENCFF757PTA_sorted_HUVEC.bam",
  'GM12891':_ENCODE_DATA_PREFIX + "ENCFF070BAN_sorted_GM12891.bam",
  'MCF-7':_ENCODE_DATA_PREFIX + "ENCFF322PZC_sorted_MCF7.bam",
  'GM12892': _ENCODE_DATA_PREFIX + "ENCFF260LKE_sorted_GM12892.bam",
  'HCT-116': _ENCODE_DATA_PREFIX + "ENCFF291HHS_sorted_HCT116.bam"
}


_TRAIN_REGIONS = [0, 2200000-1]
_VALID_REGIONS = [2200000, 2204000-1]
# 227512 for test (chr8 and chr9) for 227512 total rows
_TEST_REGIONS  = [2204000, 2204000 + 227512-1] # only up to chrs 8 and 9


# ## Functions
# 
# ### Functions for indexing genomic regions from allTFs.pos.bed file


### Load allTFs.pos regions

# read bed file line to get chr, start and end. 
# Increment by one because first line of file is blank

def getIndex(i, REGION_RANGE):
    '''
    Gets the index into allTFs.pos.bed for a given record. This will be different for train, test and valid.
    Also returns the strand (+ or -) for this record.

    Args:
        :param i: record position to index from
        :param REGION_RANGE: length 2 array containing start and end locations for this dataset in allTFs.pos.bed

    Returns:
        String of reference region of form chr:start-stop,strand

    '''
    
    # 220000 for train, 8000 for valid, 227512 for test
    REGION_LENGTH = REGION_RANGE[1]-REGION_RANGE[0]+1 # of rows for this datset in allTfs.pos file
    TOTAL_RECORDS = REGION_LENGTH * 2

    if (i < REGION_LENGTH):
        return (i + REGION_RANGE[0], "+")
    elif (i >= REGION_LENGTH and i < (TOTAL_RECORDS)):
        return (i + REGION_RANGE[0] - REGION_LENGTH, "-")
    else:
        raise Exception("index %i is out of bound for %i records" % (i, REGION_LENGTH))
    

def getRegionByIndex(i, strand = "+"):
    '''
    Gets the genomic region at line i+1 from the _DEEPSEA_GENOME_REGIONS_FILENAME file (allTFs.pos.bed)
    
    Args:
        :param i: position to index to in file (0 based)
        
    Returns:
        String of reference region of form chr:start-stop,strand
        
    '''
    
    bed_row = linecache.getline(os.path.join(_DEEPSEA_GENOME_REGIONS_FILENAME), i+1).split('\t') # file is one indexed
    region_chr = bed_row[0]
    region_start = int(bed_row[1]) - 400 # Get flanking base pairs. Deepsea only looks at middle 200 bp
    region_stop = int(bed_row[2]) + 400  # Get flanking base pairs. Deepsea only looks at middle 200 bp
    
    return GenomicInterval( region_chr, region_start, region_stop, strand=strand)


# ### Functions for getting DNase cut sites and saving results

# ### Functions for getting DNase cut sites and saving results

def saveDNaseCellTypes(data, TFPOS_REGIONS, hf):
    
    # stores dictionary of celltype: np array of dnase
    for cell in dnase_file_dict.keys(): # for all cells
        
        print("processing cell %s from file %s..." % (cell, dnase_file_dict[cell]))
        
        # read in DNase bam file
        reads = pyDNase.BAMHandler(dnase_file_dict[cell])
        cell_dnase = ss.lil_matrix((data["x"].shape[0], 1000), dtype=np.int8)
        
        # for all rows in dataset, read DNase cut sites for that region
        for row_i in range(data['x'].shape[0]): # for all rows in dataset

                # index into allTFpos.bed file to get the genomic region
                # for reading DNase
                file_index, strand = getIndex(row_i,TFPOS_REGIONS)
                region = getRegionByIndex(file_index, strand = strand)

                # get DNase
                dnase = reads[region][region.strand]
                
                # store DNase
                cell_dnase[row_i,:] = dnase
                
                if (row_i % 100000 == 0):
                    print("cell %s on iteration %i" % (cell, row_i))
                
        # write to h5 file
        hf.create_dataset(cell,data=ss.csr_matrix(cell_dnase))
        
# ## Process data

# ### Process validation data and save as .mat file

# filepath for all data

tmp = loadmat(os.path.join(deepsea_path, "valid.mat"))
data = {
    "x": tmp['validxdata'],
    "y": tmp['validdata'].T
}

with h5sparse.File( _PROCESSED_DATA_PATH + "processed_dnase_valid_sparse.h5", 'w') as h5valid:

     saveDNaseCellTypes(data, _VALID_REGIONS, h5valid)


### Process test data and save as .mat file

tmp = loadmat(os.path.join(deepsea_path, "test.mat"))
data = {
     "x": tmp['testxdata'],
     "y": tmp['testdata'].T
}
with h5sparse.File( _PROCESSED_DATA_PATH + "processed_dnase_test_sparse.h5", 'w') as h5test:
     saveDNaseCellTypes(data, _TEST_REGIONS, h5test)


# ### Process train data and save as .mat file

tmp = h5py.File(os.path.join(deepsea_path, "train.mat"), "r")
data = {
    "x": tmp["trainxdata"][()].transpose([2,1,0]),
    "y": tmp["traindata"][()]
}
with h5sparse.File( _PROCESSED_DATA_PATH + "processed_dnase_train_sparse.h5", 'w') as h5train:
    saveDNaseCellTypes(data, _TRAIN_REGIONS, h5train)

