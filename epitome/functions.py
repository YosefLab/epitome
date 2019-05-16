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
    
    def overlaps(self, other, min_bp = 1):
        return(other.chrom == self.chrom and self.overlap([self.start, self.end], [other.start, other.end]) > min_bp)
    
    def overlap(self, interval1, interval2):
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

def load_deepsea_data_allpos_file(deepsea_path):
    """
    Loads deepsea data in, concatenating train, valid, and test into 1 object.
    Each row of the matrix corresponds to the first 2431512 lines of the allpos.bed file.
    Note: Currently does not load in DNA sequence.

    """
    tmp = h5py.File(os.path.join(deepsea_path, "train.mat"))
    train_data = {
        "y": tmp["traindata"][()][:,0:2200000]
    }


    tmp = loadmat(os.path.join(deepsea_path, "valid.mat"))
    valid_data = {
        "y": tmp['validdata'].T[:,0:4000]
    }

    tmp = loadmat(os.path.join(deepsea_path, "test.mat"))
    test_data = {
        "y": tmp['testdata'].T[:,0:_TEST_REGIONS[1]-_TEST_REGIONS[0]+1] # Length of regions in allpos.bed file
    }

    data = {
        "y": np.concatenate((train_data["y"], valid_data["y"], test_data["y"]), axis=1)
    }

    return data

def save_deepsea_label_data(deepsea_path, label_output_path):
    """
    Saves just deepsea labels, easier to access and much smaller dataset that can be loaded quicker.
    
    Args:
        :param deepsea_path: path to .mat data, downloaded from script ./bin/download_deepsea_data
        :param label_output_path: new output path to save labels

    """
    if not os.path.exists(label_output_path):
        os.mkdir(label_output_path)
        print("%s Created " % label_output_path)
    
    
    tmp = h5py.File(os.path.join(deepsea_path, "train.mat"))
    train_data = {
        "y": tmp["traindata"][()]
    }


    tmp = loadmat(os.path.join(deepsea_path, "valid.mat"))
    valid_data = {
        "y": tmp['validdata'].T
    }

    tmp = loadmat(os.path.join(deepsea_path, "test.mat"))
    test_data = {
        "y": tmp['testdata'].T
    }

    valid_data = {
        "y": np.concatenate([train_data["y"][:,2200000:2400000],train_data["y"][:,4200000:4400000],valid_data["y"]], axis=1),
    }

    train_data = {
        "y": np.concatenate([train_data["y"][:,0:2200000],train_data["y"][:,2400000:4200000], test_data["y"]], axis=1),
    } 
    # save files
    
    print("saving train.mat, valid.mat and test.mat to %s" % label_output_path)
    savemat(os.path.join(label_output_path, "train.mat"), train_data)
    savemat(os.path.join(label_output_path, "valid.mat"), valid_data)
    savemat(os.path.join(label_output_path, "test.mat"), test_data)
    
def load_deepsea_label_data(deepsea_path):
    """
    Loads just deepsea labels, saved from save_deepsea_label_data function
    
    Args:
        :param deepsea_path: path to .mat data, saved by save_deepsea_label_data function
        
    :returns: train_data, valid_data, and test_data
        3 dictionaries for train, valid and test containing a 'y' matrix of labels
    """
    
    train_data = loadmat(os.path.join(deepsea_path, "train.mat"))

    valid_data = loadmat(os.path.join(deepsea_path, "valid.mat"))


    test_data = loadmat(os.path.join(deepsea_path, "test.mat"))

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



    valid_data = {
        "x": np.concatenate([train_data["x"][2200000:2400000,:,:],train_data["x"][4200000:4400000,:,:],valid_data["x"]], axis=0),
        "y": np.concatenate([train_data["y"][:,2200000:2400000],train_data["y"][:,4200000:4400000],valid_data["y"]], axis=1),
    }

    train_data = {
        "x": np.concatenate([train_data["x"][0:2200000,:,:],train_data["x"][2400000:4200000,:,:], test_data["x"]], axis=0),
        "y": np.concatenate([train_data["y"][:,0:2200000],train_data["y"][:,2400000:4200000], test_data["y"]], axis=1),
    } 

    return train_data, valid_data, test_data


def get_dnase_array_from_modified(dnase_train, dnase_valid, dnase_test, index_i, celltype, DATA_LABEL="train", num_records = 1):
    ''' This function pull respective validation and training data corresponding to
    the points chosen above.
    Args:
        :param dnase_train: h5sparse file of 1000 by n for each celltype
        :param dnase_valid: h5sparse file of 1000 by n for each celltype 
        :param dnase_test: h5sparse file of 1000 by n for each celltype 
        :param index_i: what row to index to in dataset
        :param cell_type: string of cell typex
        :param DATA_LABEL: should be train, valid or test
        :param num_records: number of records to fetch, starting at index_i
    '''
    if (DATA_LABEL == "train"):
        # from above, we concatenated [train_data["x"][0:2200000,:,:]
        # and train_data["x"][2400000:4200000,:,:]]
        if (index_i >= 0 and index_i < 2200000):
            return dnase_train[celltype][index_i:index_i+num_records].toarray()
        else:
            new_index = index_i + (2400000 - 2200000) # increment distance we removed from train set
            return dnase_train[celltype][new_index:new_index+num_records].toarray()
        
    elif (DATA_LABEL == "valid"):
        
        # from above, we concatenated [train_data["x"][2200000:2400000,:,:],
        #    train_data["x"][4200000:4400000,:,:] 
        #    valid_data["x"]], axis=0),
        if (index_i >= 0 and index_i < 2400000-2200000):
            new_index = 2200000 + index_i
            return dnase_train[celltype][new_index:new_index+num_records].toarray()
        elif (index_i >= (2400000-2200000) and index_i < (2400000-2200000) + (4400000-4200000)):
            new_index = 4200000 + index_i
            return dnase_train[celltype][new_index:new_index+num_records].toarray()
        else:
            new_index = index_i - ((2400000-2200000) + (4400000-4200000)) # between 0 to 8000
            return dnase_valid[celltype][new_index:new_index+num_records].toarray()
            
    else:
        return dnase_test[celltype][index_i:index_i+num_records].toarray()
    



def get_assays_from_feature_file(feature_path, 
                                 eligible_assays = None, 
                                 eligible_cells = None,
                                 min_cells_per_assay= 3, 
                                 min_assays_per_cell = 2):
    ''' Parses a feature name file from DeepSea. File can be found in repo at ../data/feature_name.
    Returns at matrix of cell type/assays which exist for a subset of cell types.

    Args:
        :param feature_path: location of feature_path
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
            
    # TFs are 126 to 816 and DNase is 1 to 126, TFs are 126 to 816
    # We don't want to include histone information.
    elegible_assay_indices  = np.linspace(1,815, num=815).astype(int)

    # TODO want a dictionary of assay: {list of cells}
    # then filter out assays with less than min_cells_per_assay cells
    # after this, there may be some unused cells so remove those as well
    with open(feature_path) as f:

        indexed_assays={}    # dict of {cell: {dict of indexed assays} }
        for i,l in enumerate(f):
            if i not in elegible_assay_indices: 
                continue # skip first rows and non-transcription factors

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


###############################################################################
################### Processing h5sparse files for DNase #######################

def toSparseDictionary(sparse_map, normalize): 
    ''' Converts h5 file to dictionary of sparse matrices, indexed by cell type.
    
    Args:
        :param sparse_map: map of (cell type, sparse matrix)
        :param normalize: boolean specifying whether or not to normalize the data
        
    Returns:
        dictionary of (celltype, sparsematrix)
    
    '''
    if (normalize):
        return {x[0]:x[1]/x[1].max() for x in sparse_map}
    else:
        return {x[0]:x[1] for x in sparse_map}


def toSparseIndexedDictionary(dnase_train, dnase_valid,dnase_test, DATA_LABEL, normalize = True):
    ''' Converts h5 file to dictionary of sparse matrices, indexed by cell type.
    Corrects for indices when we moved some of train into the validation set.
    
    Args:
        :param dnase_train: h5sparse file for train
        :param dnase_valid: h5sparse file for valid
        :param dnase_test: h5sparse file for test
        :param DATA_LABEL: specifies Dataset.TRAIN, valid or TEST
        :param normalize: boolean specifying whether or not to normalize the data
        
    Returns:
        dictionary of (celltype, sparsematrix)
    '''
    
    if (DATA_LABEL == Dataset.TRAIN):
        # from above, we concatenated [train_data["x"][0:2200000,:,:]
        # and train_data["x"][2400000:4200000,:,:]] 
        m = map(lambda x: (x[0], dnase_train[x[0]].value[np.r_[0:2200000,2400000:4200000]]), dnase_file_dict.items())
        return toSparseDictionary(m, normalize)
    
    elif (DATA_LABEL == Dataset.VALID):
        
        # from above, we concatenated [train_data["x"][2200000:2400000,:,:],
        #    then train_data["x"][4200000:4400000,:,:] 
        #    then valid_data["x"]], axis=0),
        m = map(lambda x: (x[0], 
                           vstack([dnase_train[x[0]].value[np.r_[2200000:2400000,4200000:4400000]],
                                   dnase_valid[x[0]].value])), dnase_file_dict.items())
        
        return toSparseDictionary(m, normalize) 
    elif (DATA_LABEL == Dataset.TEST): # test
        m = map(lambda x: (x[0], dnase_test[x[0]][:]), dnase_file_dict.items())
        return toSparseDictionary(m, normalize)
    else:
        raise
            
def get_dnase_array_from_modified_dict(dnase_train, dnase_valid, dnase_test, range_i, celltype, DATA_LABEL="train"):
    ''' This function pulls respective validation and training data corresponding to
    the points chosen above.
    Args:
        :param dnase_train: dict of 1000 by n matrix for each celltype
        :param dnase_valid: dict of 1000 by n matrrix for each celltype 
        :param dnase_test: dict of 1000 by n matrix for each celltype 
        :param range_i: what rows to index in to dataset
        :param cell_type: string of cell typex
        :param DATA_LABEL: should be train, valid or test
    Returns:
        csr_matrix sparse matrix of cut site counts
    '''
    
    if (DATA_LABEL == Dataset.TRAIN):
        return dnase_train[celltype][range_i]
    elif (DATA_LABEL == Dataset.VALID):
        return dnase_valid[celltype][range_i]
    elif (DATA_LABEL == Dataset.TEST):
        return dnase_test[celltype][range_i]
    else:
        raise
    
    
################### Parsing data from bed file ########################

def bedFile2Vector(bed_file, all_pos_file, duplicate = True):
    """
    This function takes in a bed file of peaks and converts it to a vector or 0/1s that can be 
    uses as input into a model. Each 0/1 represents a region in the train/test/validation set from DeepSEA.

    Most likely, the bed file will be the output of the IDR function, which detects peaks based on the
    reproducibility of multiple samples.

    :param: bed_file: bed file containing peaks
    :param: all_pos_file: file from DeepSEA that specified  genomic region ordering
    :param: duplicate: duplicates the strands to match deepsea's dataset. This should eventually be removed, as we do not want duplication if no DNA sequence is used. 

    :return: vector containing the concatenated 4.4 million train, validation, and test data in the order
    that we have parsed train/test. And returns a zipped of regions in the bed file and whether or not that
    peak had an associated peak in the training set.

    """

    # load in bed file and tf pos file
    positions = BedTool(all_pos_file)
    
    
    idr_peaks = []
    
    # load in peaks as a list of regions
    with open(bed_file) as f:
        content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        init_idr_peaks = list(map(lambda x: Region(x.split()[0], int(x.split()[1]), int(x.split()[2])), content))
    
    idr_peaks = list(enumerate(init_idr_peaks))
    
    # min base pair overlap to consider peak within a training region
    bp_overlap = 100

    peak_vector=np.zeros(_TOTAL_POSFILE_REGIONS)

    n_peaks = len(idr_peaks)
    
    # keeps track of which idr peaks were overlapping something in the positions file
    found = np.zeros(n_peaks)
    
    def setFound(i):
        found[i]=1

    curr_chrom = "chr1"
    idr_peaks_for_chrom = list(filter(lambda x: x[1].chrom == curr_chrom, idr_peaks))
    
    for position_i, pos in enumerate(positions):
            

        region = Region(pos.chrom, pos.start, pos.end)
        
        if (region.chrom != curr_chrom):
            curr_chrom == region.chrom
            idr_peaks_for_chrom = list(filter(lambda x: x[1].chrom == curr_chrom, idr_peaks))

        # filter in idr_peaks that overlap
        # takes >1s a while if empty
        it = filter(lambda x: x[1].overlaps(region, bp_overlap), idr_peaks_for_chrom)
    
        # if peak exists, set peak_vector to 1
        x = next(it, None)
        if (x != None):
            peak_vector[position_i] = 1
            found[x[0]] = 1
            map(lambda x: setFound(x[0]), it)
            
        # remove elements that are behind position_i
        k = 0
        while (region.greaterThan(idr_peaks[k][1])):
            k += 1
            if (k >= len(idr_peaks)):
                break
        
        del idr_peaks[0:k]
        
        # once idr_peaks is empty, return
        if (len(idr_peaks) == 0):
            break
            
        if (position_i == 10000):
            print("processing through lines in allpos bed file %i" % position_i)

            
    # filter out rows that were not used for training (defined in constants.py)
    # also, add in fake negative strands that just duplicate the sets 
    ATAC_train = peak_vector[_TRAIN_REGIONS[0]:_TRAIN_REGIONS[1]+1]
    ATAC_valid = peak_vector[_VALID_REGIONS[0]:_VALID_REGIONS[1]+1]
    ATAC_test =  peak_vector[_TEST_REGIONS[0]:_TEST_REGIONS[1]+1]

    # need to duplicate results for forward/reverse strand
    if (duplicate):
        vector = np.concatenate([ATAC_train, ATAC_train, ATAC_valid, ATAC_valid, ATAC_test, ATAC_test], axis=0)
    else:
        vector = np.concatenate([ATAC_train, ATAC_valid, ATAC_test], axis=0)

    return vector, zip(init_idr_peaks, found)
        
