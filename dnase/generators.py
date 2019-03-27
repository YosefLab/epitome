"""
Functions for data generators.
"""

######################### Original Data Generator: Only peak based #####################
def gen_from_peaks(data, 
                 label_cell_types,  # used for labels. Should be all for train/eval and subset for test
                 eval_cell_types,   # used for rotating features. Should be all - test for train/eval
                 matrix,
                 assaymap,
                 cellmap,
                 radii,
                 **kwargs):

    """
    Takes Deepsea data and calculates distance metrics from cell types whose locations
    are specified by label_cell_indices, and the other cell types in the set. Label space is only one assay.
     TODO AM 3/7/2019
    :param data: dictionary of matrices. Should have keys x and y. x contains n by 1000 rows. y contains n y 919 labels.
    :param label_cell_types: list of cell types to be rotated through and used as labels (subset of eval_cell_types)
    :param eval_cell_types: list of cell types to be used in evaluation (includes label_cell_types)
    :param matrix: matrix of celltype, assay positions
    :param assaymap: map of column assay positions in matrix
    :param cellmap: map of row cell type positions in matrix
    :param radii: radii to compute dnase distances from
    :param kwargs: kargs

    :returns: generator of data with three elements:
        1. record features
        2. record labels for a given cell type
        3. 0/1 mask of labels that have validation data. For example, if this record is for celltype A549,
        and A549 does not have data for ATF3, there will be a 0 in the position corresponding to the label space.
    """

    mode = kwargs.get("mode")
    debug = kwargs.get("debug")
    
    
    if (not isinstance(debug, bool)):
        debug = False # if not defined, do not run in debug mode
    
    
    if (not isinstance(mode, Dataset)):
        raise ValueError("mode is not a Dataset enum")
    
    print("using %s as labels for mode %s" % (label_cell_types, mode))
    
    if (mode == Dataset.TEST):
        # TODO AM 3/7/2019 drop cell types using DNase distance metric for better accuracy
        eval_cell_types = eval_cell_types.copy()
        [eval_cell_types.pop() for i in range(len(label_cell_types))]
    
    def g():
        
        if (debug): # if in debug mode, only run for 10 records.
            if (len(radii) > 0):
                range_ = range(max(radii), max(radii)+10)
            else: 
                range_ = range(0, 10)
        else:
            if (len(radii) > 0):
                range_ = range(max(radii), data["y"].shape[-1]-max(radii))
            else: 
                range_ = range(0, data["y"].shape[-1])

            
        for i in range_: # for all records
            
            for (cell) in label_cell_types: # for all cell types to be used in labels
                dnases = [] 
                
                # cells to be featurized
                feature_cells = eval_cell_types.copy()
                
                # try to remove cell if it is in the possible list of feature cell types
                try:
                    feature_cells.remove(cell) 
                except ValueError:
                    pass  # do nothing!
                
                
                feature_cell_indices_list = list(map(lambda cell: get_y_indices_for_cell(matrix, cellmap, cell), feature_cells))
                feature_cell_indices = np.array(feature_cell_indices_list).flatten()
                
                label_cell_indices = get_y_indices_for_cell(matrix, cellmap, cell)
                label_cell_indices_no_dnase = np.delete(label_cell_indices, [0])

                # Copy assay_index_no_dnase and turn into mask of 0/1 for whether data for this cell type for
                # a given label is available.
                assay_mask = np.copy(label_cell_indices_no_dnase)
                assay_mask[assay_mask == -1] = 0
                assay_mask[assay_mask > 0] = 1
                
                # get dnase indices for cell types that are going to be features
                dnase_indices = [x[0] for x in feature_cell_indices_list]
            
                for radius in radii:
                    # within the radius, fraction of places where they are both 1
                    # label_cell_index[0] == DNase location for specific cell type
                    dnase_double_positive = np.average(data["y"][dnase_indices,i-radius:i+radius+1]*
                                             data["y"][label_cell_indices[0],i-radius:i+radius+1], axis=1)
                    
                    # within the radius, fraction of places where they are both equal (0 or 1)
                    dnase_agreement = np.average(data["y"][dnase_indices,i-radius:i+radius+1]==
                                             data["y"][label_cell_indices[0],i-radius:i+radius+1], axis=1)
                    dnases.extend(dnase_double_positive)
                    dnases.extend(dnase_agreement)

                # Handle missing values 
                # Set missing values to 0. Should be handled later.
                features = data["y"][feature_cell_indices,i]

                # one hot encoding (ish). First row is 1/0 for known/unknown. second row is value.
                binding_features_n  = len(features)
                feature_n = binding_features_n + len(dnases)
                tmp = np.ones([2, feature_n])
                tmp[0,np.where(feature_cell_indices == -1)[0]] = 0 # assign UNKs to missing features
                
                tmp[1,0:binding_features_n] = features
                tmp[1,binding_features_n:]  = dnases
                
                yield tmp, \
                         data["y"][label_cell_indices_no_dnase,i], \
                         assay_mask

    return g




# as of 3/27/2019, this function is no longer being used
# def gen_from_chromatin_vector(data, y_index_vectors, dnase_indices, indices, radii, **kwargs):
#     """
#     data generator for DNase. 
    
#     :param data: dictionary of matrices. Should have keys x and y. x contains n by 1000 rows. y contains n y 919 labels.
#     :param y_index_vectors: list of vectors which the indices in the y labels that should be used. 
#     :param radii: where to calculate DNase similarity to.
    
#     :returns: generator of data
#     """
#     def g():
#         dnase_vector = kwargs["dnase_vector"]
                    
#         if (len(radii) > 0):
#             range_ = range(max(radii), data["y"].shape[-1]-max(radii))
#         else: 
#             range_ = range(0, data["y"].shape[-1])
 
#         for i in range_: # for all records
#             for y_index in y_index_vectors:
#                 dnases = [] 
#                 for radius in radii:
                        
#                     # within the radius, fraction of places where they are both 1
#                     dnase_double_positive = np.average(data["y"][dnase_indices,i-radius:i+radius+1]*
#                                              dnase_vector[i-radius:i+radius+1], axis=1)

#                     # within the radius, fraction of places where they are both equal (0 or 1)
#                     dnase_agreement = np.average(data["y"][dnase_indices,i-radius:i+radius+1]==
#                                              dnase_vector[i-radius:i+radius+1], axis=1)


#                     dnases.extend(dnase_double_positive)
#                     dnases.extend(dnase_agreement)
                    
#                 # Remove DNase from prediction indices. 
#                 # You should not predict on assays you use to calculate the distance metric.
#                 y_index_no_dnase = np.delete(y_index, [0])
#                 yield np.concatenate([data["y"][indices,i],dnases]), data["y"][y_index_no_dnase,i] 
#     return g


############################################################################################
######################## Functions for running data generators #############################
############################################################################################

# TODO AM 3/7/2019 remove this function, not really being used
def make_dataset(data,
                 label_cell_types,
                 eval_cell_types,
                 generator,
                 matrix,
                 assaymap,
                 cellmap,
                 radii=[1,3,10,30],
                **kwargs):
    
    """
    Original data generator for DNase. Takes Deepsea data and calculates distance metrics from cell types whose locations
    are specified by y_index_vector, and the other cell types in the set.
    
    :param data: dictionary of matrices. Should have keys x and y. x contains n by 1000 rows. y contains n y 919 labels.
    :param test_cell_types: list of string of all cell types to used for test.
    :param generator: which generator to use. 
    :param matrix: celltype by assay matrix holding positions of labels
    :param assaymap: map of (str assay, iloc col in matrix)
    :param cellmap: map of  (str cellname, iloc row in matrix)
    :param radii: where to calculate DNase similarity to.
    
    :returns: generator of data
    """
    g = generator(data, 
                 label_cell_types,  # used for labels. Should be all for train/eval and subset for test
                 eval_cell_types,   # used for rotating features. Should be all cell types minus test for train/eval
                 matrix,
                 assaymap,
                 cellmap,
                 radii,
                 **kwargs)

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
        output_shapes=(x.shape, y.shape, z.shape,)
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.shuffle(shuffle_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(prefetch_size)

    return y.shape, dataset.make_one_shot_iterator()

    
    
    


