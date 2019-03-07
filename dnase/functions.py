# imports 
import h5py

from pybedtools import BedTool
from scipy.io import savemat

import pandas as pd
import collections
import numpy as np
import os

from scipy.io import loadmat

######################################################
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
                                 eligible_assays = DEFAULT_ASSAYS, 
                                 eligible_cells = DEFAULT_CELLS,
                                 min_assays = 3):
    ''' Parses a feature name file from DeepSea. File can be found in repo at ../data/feature_name.
    Returns at matrix of cell type/assays which exist for a subset of cell types.
    NOTE: this changes the ordering from the previous function. Dnase is not first.

    Args:
        :param feature_path: location of feature_path
        :param factors: list of assays to filter by (ie ["CTCF", "EZH2", ..]). If None, then returns all assays.
        Note that DNase will always be included in the factors, as it is required by the method.
        :param cells: list of cells to filter by (ie ["HepG2", "GM12878", ..]). If None, then returns all cell types.
        :param min_assays: number of assays a cell type must have to be considered
    Returns
        matrix: cell type by assay matrix
        cellmap: index of cells
        assaymap: index of assays
    '''
    # if eligible_cells is specified, set min_assays to 0. We want to use all eligible_cells, no matter what.
    if (min_assays != None and eligible_cells != None):
        print("Warning: min_assays = %i and elegible_cells != None, setting min_assays to 0 to use all specified eligible cells" % (min_assays))
        min_assays = 0
    
    if (eligible_assays != None):     
        if (len(eligible_assays) + 1 < min_assays):
            raise Exception("""%s is less than the minimum assays required (%i). 
            Lower min_assays to (%i) if you plan to use only %i eligible assays""" \
                            % (eligible_assays, min_assays, len(eligible_assays)+1, len(eligible_assays)))

    # TFs are 126 to 816 and DNase is 1 to 126, TFs are 126 to 816
    # We don't want to include histone information.
    elegible_assay_indices  = np.linspace(1,815, num=815).astype(int)

    with open(feature_path) as f:

        assays = {}          # dict of (cell: assay names)
        indexed_assays={}    # dict of (cell: dict of indexed assays)
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
                if cell not in assays:
                    assays[cell] = set([assay]) # add name of assay
                    indexed_assays[cell] = {assay: i-1} # add index of assay
                else:
                    assays[cell].add(assay)
                    indexed_assays[cell][assay] = i-1



    cells = []

    # list of assays
    all_assays = []
    for lst in assays.values():
        all_assays.extend(lst)

    all_assays = collections.Counter(all_assays)
    all_assays = [(all_assays[k], k) for k in all_assays]
    sorted(all_assays, reverse=True)


    all_cells = [(len(assays[cell]), cell) for cell in assays]
    sorted(all_cells, reverse=True)

    potential_assays = []
    cells = []

    for a in assays: # where a is {cell type: [list of assays]
        if 'DNase' in assays[a] and len(assays[a]) >= min_assays:
            potential_assays.extend(assays[a])
            cells += [(len(assays[a]), a)]

    # sort cells alphabetical
    cells = sorted(cells, reverse=True, key=lambda x: x[1])
    cells = [i[1] for i in cells]
    potential_assays = collections.Counter(potential_assays)
    
    # sort assays alphabetically
    potential_assays = sorted([(potential_assays[k], k) for k in potential_assays], reverse=True, key=lambda x: x[1])
    potential_assays = [i[1] for i in potential_assays]
    
    # make sure DNase is first assay. This is because the model
    # assumes the first column specifies DNase 
    potential_assays.remove("DNase")
    potential_assays.insert(0,"DNase")

    cellmap = {cell: i for i, cell in enumerate(cells)}
    assaymap = {assay: i for i, assay in enumerate(potential_assays)}

    matrix = np.zeros((len(cellmap), len(assaymap))) - 1
    for cell in cells:
        for assay in assays[cell].intersection(potential_assays):
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
    that WE have parsed train/test.

    """

    # load in bed file and tf pos file
    positions = BedTool(all_pos_file)
    idr_peaks = BedTool(bed_file)

    # get overlaps, with overlap size 0
    c = positions.window(idr_peaks, w=0).overlap(cols=[2,3,8,9]) # cols = [start 1, end 1, start 2, end 2]
    df = c.to_dataframe()

    # only consider records that have >100bp overlap
    filtered = df[df.iloc[:,-1] > 100] # last column will contain the bp overlap 

    # get positions dataframe
    positions_df = positions.to_dataframe()

    # merge positions and regions
    regions = filtered[[0,1,2]].drop_duplicates()
    merged_df = pd.merge(positions_df, regions,  how='left', left_on=['chrom','start', 'end'], right_on = [0,1,2])

    # convery array to np array of 0/1s
    peak_vector=np.array(list(map(lambda x: float.is_integer(x), merged_df[1]))).astype(int)

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
        
    return vector
   
        