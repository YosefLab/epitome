# imports 
import h5py

######################################################
################### FUNCTIONS ########################
######################################################



################## LOADING DATA ######################
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
        "x": np.concatenate([train_data["x"][0:2200000,:,:],train_data["x"][2400000:4200000,:,:]], axis=0),
        "y": np.concatenate([train_data["y"][:,0:2200000],train_data["y"][:,2400000:4200000]], axis=1),
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
    

def get_assays_from_feature_file(feature_path='../data/feature_name'):
    ''' Parses a feature name fil from DeepSea. File can be found in repo at ../data/feature_name.
    Returns at matrix of cell type/assays which exist for a subset of cell types.
    
    Args:
        :param feature_path: location of feature_path
        
    Returns
        matrix: cell type by assay matrix
        cellmap: index of cells
        assaymap: index of assays
    '''
    
    with open(feature_path) as f:
        i = 0
        assays = {}          # dict of (cell: assay names)
        indexed_assays={}    # dict of (cell: dict of indexed assays)
        for i,l in enumerate(f):
            if i == 0:
                continue # nothing in first row

            # for example, split '8988T|DNase|None' 
            cell, assay = l.split('\t')[1].split('|')[:2]
            if assay == 'DNase':
                assays[cell] = set([assay]) # add name of assay
                indexed_assays[cell] = {assay: i-1} # add index of assay
            elif cell in assays:
                assays[cell].add(assay)
                indexed_assays[cell][assay] = i-1

    cells = []

    # best_assays
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
    best_assays = set(all_assays)


    for a in assays: # for each celltype
        # TODO Weston why are we filtering out NH here?
        if len(assays[a]) > 5 and 'DNase' in assays[a] and a[:2]!="NH":
            potential_assays.extend(assays[a])
            best_assays = best_assays.intersection(assays[a])
            cells += [(len(assays[a]), a)]

    cells = sorted(cells, reverse=True)
    cells = [i[1] for i in cells]
    potential_assays = collections.Counter(potential_assays)
    potential_assays = sorted([(potential_assays[k], k) for k in potential_assays], reverse=True)
    potential_assays = [i[1] for i in potential_assays]

    cellmap = {cell: i for i, cell in enumerate(cells)}
    assaymap = {assay: i for i, assay in enumerate(potential_assays)}
    matrix = np.zeros((len(cellmap), len(assaymap))) - 1
    for cell in cells:
        for assay in assays[cell].intersection(potential_assays):
            matrix[cellmap[cell], assaymap[assay]] = indexed_assays[cell][assay]

    matrix = matrix.astype(int)[:11,:20]
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
    
   
        