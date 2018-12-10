"""
Functions for data generators.
"""


######################### Original Data Generator: Only peak based #####################

def gen_from_peaks(data, y_index_vectors, assay_indices, dnase_indices, indices, radii, **kwargs):
    """
    Takes Deepsea data and calculates distance metrics from cell types whose locations
    are specified by y_index_vector, and the other cell types in the set. Label space is only one assay.
    This generator is used to test single vs multilabel classification performance.
    
    :param data: dictionary of matrices. Should have keys x and y. x contains n by 1000 rows. y contains n y 919 labels.
    :param y_index_vectors: list of vectors which the indices in the y labels that should be used. 
    :param assay_indices: list of assays that should be used in the label space.
    :param dnase_indices: indices for DNase for celltypes
    :param indices: indices of cell types for the feature space (does not include cell types for eval/test)
    :param radii: where to calculate DNase similarity to.
    
    :returns: generator of data
    """
    # y indices for x and assay indices for y should have the same length
    assert len(y_index_vectors) == len(assay_indices), "Length of y_index_vectors and assay_indices must be the same (# cells evaluatated)"
    
    def g():
                    
        if (len(radii) > 0):
            range_ = range(max(radii), data["y"].shape[-1]-max(radii))
        else: 
            range_ = range(0, data["y"].shape[-1])
 
        for i in range_: # for all records
            
            for (y_index, assay_index) in zip(y_index_vectors, assay_indices):
                dnases = [] 
            
                for radius in radii:
                    # within the radius, fraction of places where they are both 1
                    # y_index[0] == DNase location for specific cell type
                    dnase_double_positive = np.average(data["y"][dnase_indices,i-radius:i+radius+1]*
                                             data["y"][y_index[0],i-radius:i+radius+1], axis=1)
                    
                    # within the radius, fraction of places where they are both equal (0 or 1)
                    dnase_agreement = np.average(data["y"][dnase_indices,i-radius:i+radius+1]==
                                             data["y"][y_index[0],i-radius:i+radius+1], axis=1)
                    dnases.extend(dnase_double_positive)
                    dnases.extend(dnase_agreement)
                    
                # Remove DNase from prediction indices. 
                # You should not predict on assays you use to calculate the distance metric.
                assay_index_no_dnase = np.delete(assay_index, [0])
                yield np.concatenate([data["y"][indices,i],dnases]), data["y"][assay_index_no_dnase,i] 
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
                 validation_cell_types,
                 all_eval_cell_types,
                 generator,
                 matrix,
                 assaymap,
                 cellmap,
                 batch_size,
                 shuffle_size,
                 prefetch_size,
                 label_assays = None,
                 radii=[1,3,10,30],
                **kwargs):
    
    """
    Original data generator for DNase. Takes Deepsea data and calculates distance metrics from cell types whose locations
    are specified by y_index_vector, and the other cell types in the set.
    
    :param data: dictionary of matrices. Should have keys x and y. x contains n by 1000 rows. y contains n y 919 labels.
    :param validation_cell_types: list of string of cell types to be evaluated
    :param all_eval_cell_types: list of string of all cell types to be evaluated/tested.
    :param generator: which generator to use. 
    :param matrix: celltype by assay matrix holding positions of labels
    :param assaymap: map of (str assay, iloc col in matrix)
    :param cellmap: map of  (str cellname, iloc row in matrix)
    :param shuffle_size: shuffle data parameter
    :param prefetch_size: blocking fetch parameter
    :param label_assays: assays to evaluate in the label space. If None, set to the input space. If not None, should always  include  DNase.
    :param radii: where to calculate DNase similarity to.
    
    :returns: generator of data
    """
    
    # used in some generators for shuffling batches
    kwargs["batch_size"] = batch_size
    kwargs["matrix"] = matrix
    kwargs["all_eval_cell_types"] = all_eval_cell_types
    
    # Make sure validation_cell_types is a subset of all_eval_cell_types
    assert set(validation_cell_types) < set(all_eval_cell_types), \
        "validation_cell_types %s must be subset of all_eval_cell_types %s" % (validation_cell_types, all_eval_cell_types)
   
    
    # indices_mat is used to pull the remaining indices from cell types not used for prediction.
    # delete  all eval cell types from the matrix so we are not using them in the feature space.
    all_eval_cell_type_indices = list(map(lambda c: cellmap[c], all_eval_cell_types))
    indices_mat = np.delete(matrix, all_eval_cell_type_indices, axis=0)

    # get all feature locations for DNase for remaining cell types (just the first column in matrix)
    dnase_indices = indices_mat[:,0] # for all of the cell types (including the cell type we are evaluating)
    indices = indices_mat[indices_mat!=-1] # remaining indices for cell types not in evaluation or or test


    if (type(validation_cell_types) != list):
        validation_cell_types = [validation_cell_types]
        
    y_index_vectors = list(map(lambda cell: get_y_indices_for_cell(matrix, cellmap, cell), validation_cell_types))

    if (label_assays == None):
        g = generator(data, y_index_vectors, y_index_vectors, dnase_indices, indices, radii, **kwargs)
    else:
        # if assays are specified, extract their indices to be used as labels
        assert  label_assays[0] == "DNase", "Dnase not first item in label_assays."
        
        # list of arrays.  Each array is for an assay
        # need to first invert the matrix so cells are row matrix, then flatten with tolist()
        # results is a list of arrays, each row conatains assay indices for a unique cell type
        assay_indices = np.array(list(map(lambda assay: get_y_indices_for_assay(y_index_vectors, assaymap, assay), label_assays))).T.tolist()
        g = generator(data, y_index_vectors, assay_indices, dnase_indices, indices, radii, **kwargs)

    for x, y in g():
        break
    
    dataset = tf.data.Dataset.from_generator(
        g,
        output_types=(tf.float32,)*2,
        output_shapes=(x.shape, y.shape,)
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.shuffle(shuffle_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(prefetch_size)
    return y.shape, dataset.make_one_shot_iterator()


