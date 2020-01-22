r"""
================
Helper functions
================
.. currentmodule:: epitome.functions

.. autosummary::
  :toctree: _generate/

  load_epitome_data
  load_bed_regions
  get_assays_from_feature_file
  bed2Pyranges
  bedtools_intersect
  bedFile2Vector
  range_for_contigs
  calculate_epitome_regions
  concatenate_all_data
"""

# imports
from epitome import *
import h5py
from scipy.io import savemat

import pandas as pd
import collections
import numpy as np
import os
from collections import Counter
from itertools import groupby
from scipy.io import loadmat
from .constants import *
import scipy.sparse
import pyranges as pr

from operator import itemgetter
import gzip
import urllib
import os
import sys
import requests
import urllib
import tqdm
from zipfile import ZipFile


# to load in positions file
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

def download_and_unzip(url, dst):
    """ Downloads a url to local destination, unzips it and deletes zip.
    
    Args:
        :param url: url to download.
        :param dst: local absolute path to download data to.
    """
    dst = os.path.join(dst, os.path.basename(url))
    
    # download data if it does not exist
    if not os.path.exists(dst):

        file_size = int(urllib.request.urlopen(url).info().get('Content-Length', -1))
        if os.path.exists(dst):
            first_byte = os.path.getsize(dst)
        else:
            first_byte = 0
        if first_byte < file_size:

            header = {"Range": "bytes=%s-%s" % (first_byte, file_size)}
            pbar = tqdm.tqdm(
                total=file_size, initial=first_byte,
                unit='B', unit_scale=True, desc="Dataset not found. Downloading Epitome data to %s..." % dst)
            req = requests.get(url, headers=header, stream=True)
            with(open(dst, 'ab')) as f:
                for chunk in req.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(1024)
            pbar.close()

    if url.endswith('.zip'):
        
        # Extract zip data if it does not exist
        if not os.path.exists(dst.split('.zip')[0]):
            with ZipFile(dst, 'r') as zipObj:
               zipObj.extractall(os.path.dirname(dst))
            
            # delete old zip to free space
            os.remove(dst)
        
def load_epitome_data(data_dir=None):
    """
    Loads data processed using data/download_encode.py. This will load three sparse matrix files
    (.npz). One for train, valid (chr7) and test (chr 8 and 9). If data is not available,
    downloads Epitome data from S3.


    Args:
        :param data_dir: Directory containing train.npz, valid.npz, test.npz,
        all.pos.bed.gz file and feature_name files saved by data/download_encode.py script.
        Defaults to data in module.

    :returns: train_data, valid_data, and test_data
        3 numpy ndarrays for train, valid and test
    """
    
    if not data_dir:
        data_dir = GET_DATA_PATH()
        download_and_unzip(S3_DATA_PATH, GET_EPITOME_USER_PATH()) 

    # make sure all required files exist
    required_paths = [os.path.join(data_dir, x) for x in REQUIRED_FILES]
    assert(np.all([os.path.exists(x) for x in required_paths]))
    npz_files = list(filter(lambda x: x.endswith(".npz"), required_paths))

    sparse_matrices = [scipy.sparse.load_npz(x).toarray() for x in npz_files]
    return {Dataset.TRAIN: sparse_matrices[0], Dataset.VALID: sparse_matrices[1], Dataset.TEST: sparse_matrices[2]}

################### Parsing Deepsea Files ########################

def load_bed_regions(bedfile):
    ''' Loads Deepsea bed file (stored as .gz format), removing
    regions that have no data (the regions between valid/test
    and chromosomes X/Y).

    Args:
        :param bedfile: path to bed file.

    Returns:
        :return list of genomic Regions the size of train/valid/test data.
    '''
    
    with gzip.open(bedfile, 'r') as f:
            liPositions = f.readlines()

    def fromString(x):
        tmp = x.decode("utf-8").split('\t')
        return Region(tmp[0], int(tmp[1]), int(tmp[2]))

    return list(map(lambda x: fromString(x), liPositions))


def list_assays(feature_name_file = None):
    
    ''' Parses a feature name file from DeepSea. File can be found in repo at ../data/feature_name.
    Returns at matrix of cell type/assays which exist for a subset of cell types.

    Args:
        :param: feature_name_file. Path to file containing cell, ChIP metadata. Defaults to module data feature file.

    Returns:
        assays: list of assay names
    '''
    if not feature_name_file:
        feature_name_file = os.path.join(DATA_PATH, FEATURE_NAME_FILE)    

    with open(feature_name_file) as f:

        assays=[]    # dict of {cell: {dict of indexed assays} }
        for i,l in enumerate(f):
            if (i == 0):
                continue

            # for example, split '8988T|DNase|None'
            _, assay = l.split('\t')[1].split('|')[:2]
            assays.append(assay)

    return assays

    

def get_assays_from_feature_file(feature_name_file = None,
                                 eligible_assays = None,
                                 eligible_cells = None,
                                 min_cells_per_assay= 3,
                                 min_assays_per_cell = 2):
    ''' Parses a feature name file. File can be found in repo at ../data/feature_name.
    Returns at matrix of cell type/assays which exist for a subset of cell types.

    Args:
        :param: feature_name_file. Path to file containing cell, ChIP metadata. Defaults to module data feature file.
        :param eligible_assays: list of assays to filter by (ie ["CTCF", "EZH2", ..]). If None, then returns all assays.
        Note that DNase will always be included in the factors, as it is required by Epitome.
        :param eligible_cells: list of cells to filter by (ie ["HepG2", "GM12878", ..]). If None, then returns all cell types.
        :param min_cells_per_assay: number of cell types an assay must have to be considered
        :param min_assays_per_cell: number of assays a cell type must have to be considered. Includes DNase.

    Returns:
        matrix: cell type by assay matrix
        cellmap: index of cells
        assaymap: index of assays
    '''
    
    if not feature_name_file:
        feature_name_file = os.path.join(DATA_PATH, FEATURE_NAME_FILE)

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


    # Want a dictionary of assay: {list of cells}
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


def bed2Pyranges(bed_file):
    """
    Loads bed file in as a pyranges object.
    Preserves ordering of bed lines.

    Args:
        :param bed_file: absolute path to bed file
    Returns:
        indexed pyranges object
    """
    # just get chromosome location (first three columns)
    p = pd.read_csv(bed_file, sep='\t',header=None)[[0,1,2]]

    p['idx']=p.index
    p.columns = ['Chromosome', 'Start','End','idx']
    return pr.PyRanges(p).sort()


def bedtools_intersect(file_triple):
    """
    Runs intersection between 2 bed files and returns a vector of 0/1s
    indicating absense or presense of overlap.

    Args:
        :param file_triple: triple of (bed_file_1, bed_file_2, boolean).
            bed_file_1: bed file to run intersection against.
            bed_file_2: bed file to check for overlaps with bed_file_1.
            boolean: boolean determines wheather to return
            original peaks from bed_file_1.

    Returns:
        tuple of (bed_file_1 peaks, vector of 0/1s) whose length is len(bed_file_1).
        1s in vector indicate overlap of bed_file_1 and bed_file_2).

    """
    bed1 = bed2Pyranges(file_triple[0])
    bed2 = bed2Pyranges(file_triple[1])

    res = bed1.join(bed2, how='left')
    overlap_vector = np.zeros(len(bed1),dtype=bool)

    # get regions with overlap and set to 1
    res_df = res.df
    if not res_df.empty: # throws error if empty because no columns
        overlap_vector[res_df[res_df['Start_b'] != -1]['idx']] = 1

    if (file_triple[2]):
        # for some reason chaining takes a lot longer, so we run ops separately.
        t1 = bed1.df.sort_values(by='idx')[['Chromosome','Start','End']]
        t2 = t1.values
        t3 = t2.tolist()
        return (list(map(lambda x: Region(x[0],x[1],x[2]), t3)), overlap_vector)
    else:
        return (None, overlap_vector)

def bedFile2Vector(bed_file, allpos_bed_file):
    """
    This function takes in a bed file of peaks and converts it to a vector or 0/1s that can be
    uses as input into an Epitome model. Each 0/1 represents a region in the train/test/validation set from DeepSEA.

    Most likely, the bed file will be the output of the IDR function, which detects peaks based on the
    reproducibility of multiple samples.

    Args:
        :param bed_file: bed file containing peaks
        :param allpos_bed_file: bed file containing all positions in the dataset

    Returns:
        :return: tuple (numpy_train_array, (bed_peaks, numpy_bed_array).
        numpy_train_array: boolean numpy array indicating overlap of training data with peak file (length of training data).
        bed_peaks: a list of intervals loaded from bed_file.
        numpy_bed_array: boolean numpy array indicating presence or absense of each bed_peak region in the training dataset.
    """

    bed_files = [(allpos_bed_file, bed_file, False), (bed_file, allpos_bed_file, True)]
    pool = multiprocessing.Pool(processes=2)
    results = pool.map(bedtools_intersect, bed_files)

    pool.close()
    pool.join()

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


##################### Functions for parsing allpos.bed file ################
def range_for_contigs(all_regions_file):
    """
    Traverses through feature_name_file to get contig ranges.

    Args:
        :param all_regions_file: path to bed file of genomic regions

    Returns:
        list of contigs and their start/end position all_regions_file
    """
    with gzip.open(all_regions_file,'rt') as f:

        contigs = {}
        this_contig = "placeholder"
        this_range = [0,0]
        for num, line in enumerate(f):
            contig = line.split("\t")[0]

            if (this_contig != contig):
                # update last contig and save information
                this_range[1] = num
                contigs[this_contig] = this_range

                # reset contig to new
                this_contig = contig
                this_range = [0,0]
                this_range[0] = num
            else:
                continue

        # add last contig
        this_range[1] = num + 1
        contigs[this_contig] = this_range

        del contigs["placeholder"]

        return contigs

def calculate_epitome_regions(all_regions_file):
    """
    Gets line numbers for train/valid/test boundaries.
    Assumes that chr7,8,9 are all in a row (only true if sex chrs are removed).


    Args:
        :param all_pos_file: bed file containing a row for each genomic region.

    Returns:
        triple of train,valid,test regions
    """

    contig_ranges = range_for_contigs(all_regions_file)

    EPITOME_VALID_REGIONS = contig_ranges["chr7"]
    # 227512 for test (chr8 and chr9)
    chr8_start = contig_ranges["chr8"][0]
    chr9_end = contig_ranges["chr9"][1]
    EPITOME_TEST_REGIONS  = [chr8_start, chr9_end] # chr 8 and 9

    EPITOME_TRAIN_REGIONS = [[0,EPITOME_VALID_REGIONS[0]],
                             [EPITOME_TEST_REGIONS[1],contig_ranges["chr21"][1]]]


    return(EPITOME_TRAIN_REGIONS, EPITOME_VALID_REGIONS, EPITOME_TEST_REGIONS)


def concatenate_all_data(data, region_file):
    """
    Puts data back in correct order to be indexed by the allpos file.

    Args:
        :param data: data dictionary of train, valid and test
        :param regions_file: bed file containg regions of train, valid and test.

    Returns:
        np matrix of concatenated data
    """

    # Get chr6 cutoff. takes about 3s.
    chr6_end = range_for_contigs(region_file)['chr6'][1]
    return np.concatenate([data[Dataset.TRAIN][:,0:chr6_end], # chr 1-6, range_for_contigs is 1 based
                           data[Dataset.VALID], # chr7
                           data[Dataset.TEST], # chr 8 and 9
                           data[Dataset.TRAIN][:,chr6_end:]],axis=1) # all the rest of the chromosomes
