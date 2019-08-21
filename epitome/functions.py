# imports 
import h5py

from pybedtools import BedTool
from scipy.io import savemat

import pandas as pd
import collections
import numpy as np
import os
from collections import Counter
from itertools import groupby
from scipy.io import loadmat
from numba import cuda
from .constants import * 
import scipy.sparse

from operator import itemgetter
import gzip

# to load in positions file
import pybedtools
import multiprocessing

################### CLASSES ##########################
# TODO move to separate file
class Region:
    """Genomic Region"""

    def __init__(self, chrom, start, end):
        self.chrom = chrom
        self.start = start
        self.end = end
        
    def __str__(self):
        return("%s:%d-%d" % (self.chrom, self.start, self.end))
    
    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, Region):
            return self.chrom == other.chrom and self.start == other.start and self.end == other.end
        return False

    def overlaps(self, other, min_bp = 1):
        #assert(type(other) == Region), "overlaps analysis must take Region as input but got %s" % type(other)
        return(other.chrom == self.chrom and Region.overlap([self.start, self.end], [other.start, other.end]) > min_bp)
    
    @staticmethod
    def overlap(interval1, interval2):
        """
        Computes overlap between two intervals
        """
        if interval2[0] <= interval1[0] <= interval2[1]:
            start = interval1[0]
        elif interval1[0] <= interval2[0] <= interval1[1]:
            start = interval2[0]
        else:
            return 0

        if interval2[0] <= interval1[1] <= interval2[1]:
            end = interval1[1]
        elif interval1[0] <= interval2[1] <= interval1[1]:
            end = interval2[1]
        else:
            return 0

        return(end - start)

    def greaterThan(self, other):
        assert(type(other) == Region), "overlaps analysis must take Region as input but got %s" % type(other)
        return(self.chrom > other.chrom or ( self.chrom == other.chrom and self.start > other.end ))


################### FUNCTIONS ########################
######################################################


################## HELPER FUNCTIONS ##################

def release_gpu_resources():
    cuda.select_device(0)
    cuda.close()
    
def get_y_indices_for_cell(matrix, cellmap, cell):
    """
    Gets indices for a cell.
    
    :param matrix: cell type by assay matrix with locations 
    label space.
    :param cellmap: map of celltype to iloc row in matrix
    :param cell: str cell name
    
    :return locations of indices for the cell name specified
    """

    return np.copy(matrix[cellmap[cell]])


def get_y_indices_for_assay(arrays, assaymap, assay):
    """
    Gets indices for a assay.
    
    :param matrix: cell type by assay matrix with locations 
    label space.
    :param cellmap: map of celltype to iloc row in matrix
    :param cell: str cell name
    
    :return locations of indices for the cell name specified
    """
    # get  column for this assay
    matrix = output = np.array(arrays)
    return np.copy(matrix[:,assaymap[assay]])

def get_missing_indices_for_cell(matrix, cellmap, cell):
    """
    Gets indices of missing factors for a given cell type.
    
    :param matrix: cell type by assay matrix with locations 
    label space.
    :param cellmap: map of celltype to iloc row in matrix
    :param cell: str cell name
    
    :return indices where this cell has no data (==-1)
    """
    
    indices = get_y_indices_for_cell(matrix, cellmap, cell)
    return np.where(indices == -1)[0]
    

################## LOADING DATA ######################

def save_deepsea_label_data(deepsea_path, label_output_path):
    """
    Saves just deepsea labels, easier to access and much smaller dataset that can be loaded quicker.
    
    Args:
        :param deepsea_path: path to .mat data, downloaded from script ./bin/download_deepsea_data
        :param label_output_path: new output path to save labels. saves as numpy files.

    """
    if not os.path.exists(label_output_path):
        os.mkdir(label_output_path)
        print("%s Created " % label_output_path)
    
    tmp = h5py.File(os.path.join(deepsea_path, "train.mat"))
    train_data = np.matrix(tmp["traindata"][:,0:int(tmp["traindata"].shape[1]/2)])

    tmp = loadmat(os.path.join(deepsea_path, "valid.mat"))
    valid_data = np.matrix(tmp['validdata'][0:int(tmp['validdata'].shape[0]/2),:].T)

    tmp = loadmat(os.path.join(deepsea_path, "test.mat"))
    test_data = np.matrix(tmp['testdata'][0:int(tmp['testdata'].shape[0]/2),:].T)
    
    print(train_data.shape, valid_data.shape, test_data.shape)

    # save files
    print("saving train.npy, valid.npy and test.npy to %s" % label_output_path)
    np.save(os.path.join(label_output_path, "train"), train_data)
    np.save(os.path.join(label_output_path, "valid"), valid_data)
    np.save(os.path.join(label_output_path, "test"), test_data)


def load_deepsea_label_data(deepsea_path):
    """
    Loads just deepsea labels, saved from save_deepsea_label_data function
    
    Args:
        :param deepsea_path: path to .npy data, saved by save_deepsea_label_data function
        
    :returns: train_data, valid_data, and test_data
        3 numpy ndarrays for train, valid and test
    """
    
    train_data = np.load(os.path.join(deepsea_path, "train.npy"))

    valid_data = np.load(os.path.join(deepsea_path, "valid.npy"))


    test_data = np.load(os.path.join(deepsea_path, "test.npy"))

    return train_data, valid_data, test_data


def load_deepsea_data(deepsea_path):
    tmp = h5py.File(os.path.join(deepsea_path, "train.mat"))
    train_data = {
        "x": tmp["trainxdata"][()].transpose([2,1,0]),
        "y": tmp["traindata"][()]
    }

    tmp = loadmat(os.path.join(deepsea_path, "valid.mat"))
    valid_data = {
        "x": tmp['validxdata'],
        "y": tmp['validdata'].T
    }

    tmp = loadmat(os.path.join(deepsea_path, "test.mat"))
    test_data = {
        "x": tmp['testxdata'],
        "y": tmp['testdata'].T
    }

    return train_data, valid_data, test_data


def load_epitome_data(data_dir):
    """
    Loads data processed using data/download_encode.py. This will load three sparse matrix files 
    (.npz). One for train, valid (chr7) and test (chr 8 and 8).
    
    Takes ~10 seconds to load data.
    
    Args:
        :param data_dir: Directory containing train.npz, valid.npz, test.npz, 
        all.pos.bed file and feature_name files saved by data/download_encode.py script.
        
    :returns: train_data, valid_data, and test_data
        3 numpy ndarrays for train, valid and test
    """
    
    # make sure all required files exist
    required_files = ["all.pos.bed","train.npz","valid.npz", "feature_name","test.npz"]
    required_paths = [os.path.join(data_dir, x) for x in required_files]
    assert(np.all([os.path.exists(x) for x in required_paths]))
    npz_files = list(filter(lambda x: x.endswith(".npz"), required_paths))
    
    sparse_matrices = [scipy.sparse.load_npz(x).toarray() for x in npz_files]
    return sparse_matrices

def get_epitome_indices_deepsea_validation(epitome_allpos_file):
    """ Gets new indices matching DeepSEA's validation set (chr7: 30508800-35296600)
    Used to compare performance of Epitome model on DeepSEA's data vs processed data.
    
    Args:
        :param epitome_allpos_file: path to Epitome's allpos.bed file
        
    Returns:
        list of indices in Epitome's dataset that correspond to DeepSEA's
        
    """
    deepsea_valid_start = 30508800
    deepsea_valid_stop = 35296600 # start of last region
    
    with open(epitome_allpos_file) as f:
        lines = f.readlines()
        
    def split_line(f):
        split = f.split("\t")
        return (split[0], int(split[1]), int(split[2]))
                          
    filtered = enumerate([split_line(line) for line in lines if line.startswith("chr7")])
    return [i[0] for i in filtered if i[1][1] >= deepsea_valid_start and i[1][1] <= deepsea_valid_stop]

################### Parsing Deepsea Files ########################

def load_bed_regions(bedfile):
    ''' Loads Deepsea bed file (stored as .gz format), removing
    regions that have no data (the regions between valid/test
    and chromosomes X/Y). 
    
    :return list of genomic Regions the size of train/valid/test data. 
    '''
    
    with gzip.open(bedfile, 'r') as f:
            liPositions = f.readlines()

    def fromString(x):
        tmp = x.decode("utf-8").split('\t')
        return Region(tmp[0], int(tmp[1]), int(tmp[2]))

    return list(map(lambda x: fromString(x), liPositions))

def get_assays_from_feature_file(feature_name_file, 
                                 eligible_assays = None, 
                                 eligible_cells = None,
                                 min_cells_per_assay= 3, 
                                 min_assays_per_cell = 2):
    ''' Parses a feature name file from DeepSea. File can be found in repo at ../data/feature_name.
    Returns at matrix of cell type/assays which exist for a subset of cell types.

    Args:
        :param eligible_assays: list of assays to filter by (ie ["CTCF", "EZH2", ..]). If None, then returns all assays.
        Note that DNase will always be included in the factors, as it is required by the method.
        :param eligible_cells: list of cells to filter by (ie ["HepG2", "GM12878", ..]). If None, then returns all cell types.
        :param min_cells_per_assay: number of cell types an assay must have to be considered
        :param min_assays_per_cell: number of assays a cell type must have to be considered. Includes DNase.
    Returns
        matrix: cell type by assay matrix
        cellmap: index of cells
        assaymap: index of assays
    '''
    
    # check argument validity
    if (min_assays_per_cell < 2):     
         print("Warning: min_assays_per_cell should not be < 2 (this means it only has DNase) but was set to %i" % min_assays_per_cell)
         
    
    if (min_cells_per_assay < 2):     
         print("Warning: min_cells_per_assay should not be < 2 (this means you may only see it in test) but was set to %i" % min_cells_per_assay)
         
    if (eligible_assays != None):     
        if (len(eligible_assays) + 1 < min_assays_per_cell):
            raise Exception("""%s is less than the minimum assays required (%i). 
            Lower min_assays_per_cell to (%i) if you plan to use only %i eligible assays""" \
                            % (eligible_assays, min_assays_per_cell, len(eligible_assays)+1, len(eligible_assays)))

    if (eligible_cells != None):     
        if (len(eligible_cells) + 1 < min_cells_per_assay):
            raise Exception("""%s is less than the minimum cells required (%i). 
            Lower min_cells_per_assay to (%i) if you plan to use only %i eligible cells""" \
                            % (eligible_cells, min_cells_per_assay, len(eligible_cells)+1, len(eligible_cells)))
            

    # TODO want a dictionary of assay: {list of cells}
    # then filter out assays with less than min_cells_per_assay cells
    # after this, there may be some unused cells so remove those as well
    with open(feature_name_file) as f:

        indexed_assays={}    # dict of {cell: {dict of indexed assays} }
        for i,l in enumerate(f):
            if (i == 0):
                continue

            # for example, split '8988T|DNase|None' 
            cell, assay = l.split('\t')[1].split('|')[:2]

            # check if cell and assay is valid
            valid_cell = (eligible_cells == None) or (cell in eligible_cells) 
            valid_assay = (eligible_assays == None) or (assay in eligible_assays) or (assay == "DNase")

            # if cell and assay is valid, add it in
            if valid_cell and valid_assay:
                if cell not in indexed_assays:
                    indexed_assays[cell] = {assay: i-1} # add index of assay
                else:
                    indexed_assays[cell][assay] = i-1



    # finally filter out cell types with < min_assays_per_cell and have DNase
    indexed_assays = {k: v for k, v in indexed_assays.items() if 'DNase' in v.keys() and len(v) >= min_assays_per_cell}

    # make flatten list of assays from cells
    tmp = [list(v) for k, v in indexed_assays.items()]
    tmp = [item for sublist in tmp for item in sublist]

    # list of assays that meet min_cell criteria
    valid_assays = {k:v for k, v in Counter(tmp).items() if v >= min_cells_per_assay}
    
    # remove invalid assays from indexed_assays
    for key, values in indexed_assays.items():
        
        # remove assays that do not mean min_cell criteria
        new_v = {k: v for k, v in values.items() if k in valid_assays.keys()}
        indexed_assays[key] = new_v 
    
    potential_assays = valid_assays.keys()
    cells = indexed_assays.keys()

    # sort cells alphabetical
    cells = sorted(cells, reverse=True)
    
    # sort assays alphabetically
    potential_assays = sorted(potential_assays, reverse=True)
    
    # make sure DNase is first assay. This is because the model
    # assumes the first column specifies DNase 
    potential_assays.remove("DNase")
    potential_assays.insert(0,"DNase")

    cellmap = {cell: i for i, cell in enumerate(cells)}
    assaymap = {assay: i for i, assay in enumerate(potential_assays)}

    matrix = np.zeros((len(cellmap), len(assaymap))) - 1
    for cell in cells:
        for assay, _ in indexed_assays[cell].items():
            matrix[cellmap[cell], assaymap[assay]] = indexed_assays[cell][assay]

    matrix = matrix.astype(int) 
    return matrix, cellmap, assaymap

    
################### Parsing data from bed file ########################



def bedtools_intersect(file_tuple):
    return_peaks = file_tuple[2]
    bed = pybedtools.BedTool(file_tuple[0])
    res =  bed.intersect(file_tuple[1], c=True)
    overlap_vector = np.array(list(map(lambda x: int(x.fields[-1])>0, res)))
    
    l = list(bed) if return_peaks else None
    return (l, overlap_vector)
    
def bedFile2Vector(bed_file, allpos_bed_file):
    """
    This function takes in a bed file of peaks and converts it to a vector or 0/1s that can be 
    uses as input into an Epitome model. Each 0/1 represents a region in the train/test/validation set from DeepSEA.
    
    Takes 30 seconds for ~40,000 peak file

    Most likely, the bed file will be the output of the IDR function, which detects peaks based on the
    reproducibility of multiple samples.

    :param bed_file: bed file containing peaks
    :param allpos_bed_file: bed file containing all positions in the dataset

    :return: tuple (numpy_train_array, (bed_peaks, numpy_bed_array). 
    numpy_train_array: boolean numpy array indicating overlap of training data with peak file (length of training data). 
    bed_peaks: a list of intervals loaded from bed_file. 
    numpy_bed_array: boolean numpy array indicating presence or absense of each bed_peak region in the training dataset.
    """

    bed_files = [(allpos_bed_file, bed_file, False), (bed_file, allpos_bed_file, True)]
    pool = multiprocessing.Pool(processes=2)
    results = pool.map(bedtools_intersect, bed_files)
    
    bed_file_peaks = list(zip(results[1][0], results[1][1]))
    
    return (results[0][1], results[1])


def indices_for_weighted_resample(data, n,  matrix, cellmap, assaymap, weights = None):
    """
    Selects n rows from data that have the greatest number of labels (can be weighted)
    Returns indices to these rows.
    
    :param data: data matrix with shape (factors, records)
    :param n: number or rows to sample
    :param matrix: cell type by assay position matrix
    :param cellmap dict of cells and row positions in matrix
    :param assaymap: dict of assays and column positions in matrix
    :param weights: Optional vector of weights whos length = # factors (1 weight for each factor).
    The greater the weight, the more the positives for this factor matters.
    """
    
    # only take rows that will be used in set
    # drop DNase from indices in assaymap first
    selected_assays = list(assaymap.values())[1:]
    indices = matrix[list(cellmap.values())][:,selected_assays].flatten()

    # set missing assay/cell combinations to -1 
    t1 = data[indices, :]
    t1[np.where(indices < 0)[0],:] = 0
    
    # sum over each factor for each record
    sums = np.sum(np.reshape(t1, (len(selected_assays), len(cellmap), t1.shape[1])), axis=1) 

    if (weights is not None):
        weights = np.reshape(weights, (weights.shape[0],1)) # reshape so multiply works
        probs = np.sum(sums * weights, axis = 0)
        probs = probs/np.sum(probs)
    else:
        # simple sum over recoreds. Weights records with more positive
        # samples higher for random sampling.
        probs = np.sum(sums, axis=0)
        probs = (probs)/np.sum(probs)
        
    # TODO assign equal probs to non-zero weights
    probs[probs != 0] = 1/probs[probs != 0].shape[0]
    
    radius = 20
    
    n = int(n / radius)
    data_count = data.shape[1]
    
    # sample by probabilities. not sorted.
    choice = np.random.choice(np.arange(0, data_count), n, p = probs)
    
    func_ = lambda x: np.arange(x - radius/2, x + radius/2)
    surrounding = np.unique(list(map(func_, choice)))
    return surrounding[(surrounding > 0) & (surrounding < data_count)].astype(int)
