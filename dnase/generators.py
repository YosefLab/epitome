"""
Functions for data generators.
"""

######################### Original Data Generator: Only peak based #####################
def gen_from_peaks(data, 
                    label_cell_indices, 
                    assay_indices, 
                    dnase_indices, 
                    feature_assay_indices, 
                    radii, 
                    **kwargs):
    """
    Takes Deepsea data and calculates distance metrics from cell types whose locations
    are specified by label_cell_indices, and the other cell types in the set. Label space is only one assay.
    
    :param data: dictionary of matrices. Should have keys x and y. x contains n by 1000 rows. y contains n y 919 labels.
    :param label_cell_indices: list of vectors which are the indices in the labels that should be used for each eval cell type.
    :param assay_indices: list of assays that should be used in the label space. 1 per test/eval cell type.
    :param dnase_indices: indices for DNase for celltypes
    :param feature_assay_indices: indices of assays for train cell types for the feature space (does not include cell types for eval/test)
    :param radii: where to calculate DNase similarity to.
    
    :returns: generator of data with three elements:
        1. record features
        2. record labels for a given cell type
        3. 0/1 mask of labels that have validation data. For example, if this record is for celltype A549,
        and A549 does not have data for ATF3, there will be a 0 in the position corresponding to the label space.
    """
    
    # y indices for x and assay indices for y should have the same length
    assert len(label_cell_indices) == len(assay_indices), "Length of label_cell_indices and assay_indices must be the same (# cells evaluatated)"
    
    mode = kwargs["mode"]
    
    # indices that we want to use in the generator
    # If none, then  will use the whole dataset for the generator
    record_indices = None
    if (mode == Dataset.TRAIN):
        record_indices = kwargs.get("train_record_indices")
    elif (mode == Dataset.VALID):
        record_indices = kwargs.get("valid_record_indices")
    ## shouldn't subset for test!
    
    def g():
                    
        if (len(radii) > 0):
            range_ = range(max(radii), data["y"].shape[-1]-max(radii))
        else: 
            range_ = range(0, data["y"].shape[-1])
            
        for i in range_: # for all records
            # check if you you actually want to add this record to the generator
            if (hasattr(record_indices, 'shape')):
                if (i not in record_indices): 
                    continue
            
            # label_cell_index and assay_index are the same unless sometimes?
            for (label_cell_index, assay_index) in zip(label_cell_indices, assay_indices):
                dnases = [] 
            
                for radius in radii:
                    # within the radius, fraction of places where they are both 1
                    # label_cell_index[0] == DNase location for specific cell type
                    dnase_double_positive = np.average(data["y"][dnase_indices,i-radius:i+radius+1]*
                                             data["y"][label_cell_index[0],i-radius:i+radius+1], axis=1)
                    
                    # within the radius, fraction of places where they are both equal (0 or 1)
                    dnase_agreement = np.average(data["y"][dnase_indices,i-radius:i+radius+1]==
                                             data["y"][label_cell_index[0],i-radius:i+radius+1], axis=1)
                    dnases.extend(dnase_double_positive)
                    dnases.extend(dnase_agreement)
                
                # Remove DNase from prediction indices. 
                # You should not predict on assays you use to calculate the distance metric.
                assay_index_no_dnase = np.delete(assay_index, [0])
                
                # Copy assay_index_no_dnase and turn into mask of 0/1 for whether data for this cell type for
                # a given label is available.
                assay_mask = np.copy(assay_index_no_dnase)
                assay_mask[assay_mask == -1] = 0
                assay_mask[assay_mask > 0] = 1
                
                # AM 3/6/2019 subsampling was hurting accuracy?
                if (mode == Dataset.TRAIN) & (len(assay_index_no_dnase) > 1): # only sample for learning
                    if (data["y"][feature_assay_indices,i].sum() != 0) | (data["y"][assay_index_no_dnase,i].sum() != 0):
                        yield np.concatenate([data["y"][feature_assay_indices,i],dnases]), data["y"][assay_index_no_dnase,i], assay_mask
      
                else: 
                    yield np.concatenate([data["y"][feature_assay_indices,i],dnases]), data["y"][assay_index_no_dnase,i], assay_mask
    return g


def gen_from_chromatin_vector(data, y_index_vectors, dnase_indices, indices, radii, **kwargs):
    """
    data generator for DNase. 
    
    :param data: dictionary of matrices. Should have keys x and y. x contains n by 1000 rows. y contains n y 919 labels.
    :param y_index_vectors: list of vectors which the indices in the y labels that should be used. 
    :param radii: where to calculate DNase similarity to.
    
    :returns: generator of data
    """
    def g():
        dnase_vector = kwargs["dnase_vector"]
                    
        if (len(radii) > 0):
            range_ = range(max(radii), data["y"].shape[-1]-max(radii))
        else: 
            range_ = range(0, data["y"].shape[-1])
 
        for i in range_: # for all records
            for y_index in y_index_vectors:
                dnases = [] 
                for radius in radii:
                        
                    # within the radius, fraction of places where they are both 1
                    dnase_double_positive = np.average(data["y"][dnase_indices,i-radius:i+radius+1]*
                                             dnase_vector[i-radius:i+radius+1], axis=1)

                    # within the radius, fraction of places where they are both equal (0 or 1)
                    dnase_agreement = np.average(data["y"][dnase_indices,i-radius:i+radius+1]==
                                             dnase_vector[i-radius:i+radius+1], axis=1)


                    dnases.extend(dnase_double_positive)
                    dnases.extend(dnase_agreement)
                    
                # Remove DNase from prediction indices. 
                # You should not predict on assays you use to calculate the distance metric.
                y_index_no_dnase = np.delete(y_index, [0])
                yield np.concatenate([data["y"][indices,i],dnases]), data["y"][y_index_no_dnase,i] 
    return g


############################################################################################
######################## Functions for running data generators #############################
############################################################################################

def make_dataset(data,
                 label_cell_types,
                 all_eval_cell_types,
                 generator,
                 matrix,
                 assaymap,
                 cellmap,
                 label_assays = None,
                 radii=[1,3,10,30],
                **kwargs):
    
    """
    Original data generator for DNase. Takes Deepsea data and calculates distance metrics from cell types whose locations
    are specified by y_index_vector, and the other cell types in the set.
    
    :param data: dictionary of matrices. Should have keys x and y. x contains n by 1000 rows. y contains n y 919 labels.
    :param label_cell_types: list of string of cell types to used for labels
    :param all_eval_cell_types: list of string of all cell types to be evaluated/tested.
    :param generator: which generator to use. 
    :param matrix: celltype by assay matrix holding positions of labels
    :param assaymap: map of (str assay, iloc col in matrix)
    :param cellmap: map of  (str cellname, iloc row in matrix)
    :param label_assays: assays to evaluate in the label space. If None, set to the input space. If not None, should always  include  DNase.
    :param radii: where to calculate DNase similarity to.
    
    :returns: generator of data
    """
    
    # used in some generators for shuffling batches
    # batch sized is used for metalearning
    
    # AM TODO TRANSFER TO METALEARNING CODE    
    batch_size = kwargs.get('batch_size',1)
    
    kwargs["matrix"] = matrix
    kwargs["all_eval_cell_types"] = all_eval_cell_types
    
    # Make sure label_cell_types is a subset of all_eval_cell_types
    assert set(label_cell_types) < set(all_eval_cell_types), \
        "label_cell_types %s must be subset of all_eval_cell_types %s" % (label_cell_types, all_eval_cell_types)
   
    
    # indices_mat is used to pull the remaining indices from cell types not used for prediction.
    # delete all eval cell types from the matrix so we are not using them in the feature space.
    all_eval_cell_type_indices = list(map(lambda c: cellmap[c], all_eval_cell_types))
    indices_mat = np.delete(matrix, all_eval_cell_type_indices, axis=0)

    # get all feature locations for DNase for remaining cell types (just the first column in matrix)
    dnase_indices = indices_mat[:,0] # for all of the cell types (including the cell type we are evaluating)
    feature_assay_indices = indices_mat[indices_mat!=-1] # remaining indices for cell types not in evaluation or test


    if (type(label_cell_types) != list):
        label_cell_types = [label_cell_types]
        
    # list of indices for each label cell type
    label_cell_indices = list(map(lambda cell: get_y_indices_for_cell(matrix, cellmap, cell), label_cell_types))

    if (label_assays == None):
        print("no label assays")
        g = generator(data, label_cell_indices, label_cell_indices, dnase_indices, feature_assay_indices, radii, **kwargs)
    else:
        # if assays are specified, extract their indices to be used as labels
        assert  label_assays[0] == "DNase", "Dnase not first item in label_assays."
        
        # list of arrays.  Each array is for an assay
        # need to first invert the matrix so cells are row matrix, then flatten with tolist()
        # results is a list of arrays, each row conatains assay indices for a unique cell type
        assay_indices = np.array(list(map(lambda assay: get_y_indices_for_assay(label_cell_indices, assaymap, assay), label_assays))).T.tolist()
        
        g = generator(data, label_cell_indices, assay_indices, dnase_indices, feature_assay_indices, radii, **kwargs)
        
    return g


def generator_to_one_shot_iterator(g, batch_size, shuffle_size, prefetch_size):
    """
    Generates a one shot iterator from a data generator.
    
    :param g: data generator
    :param batch_size: number of elements in generator to combine into a single batch
    :param shuffle_size: number of elements from the  generator fromw which the new dataset will shuffle
    :param prefetch_size: maximum number of elements that will be buffered  when prefetching
    :param radii: where to calculate DNase similarity to.
    
    :returns: tuple of (label shape, one shot iterator)
    """

    for x, y, z in g():
        break
    
    dataset = tf.data.Dataset.from_generator(
        g,
        output_types=(tf.float32,)*3, # 3 = features, labels, and missing indices
        output_shapes=(x.shape, y.shape,z.shape,)
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.shuffle(shuffle_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(prefetch_size)

    return y.shape, dataset.make_one_shot_iterator()

    
    
    


