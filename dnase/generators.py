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
    are specified by label_cell_indices, and the other cell types in the set. Label space is only one cell type.
     TODO AM 3/7/2019
    :param data: dictionary of matrices. Should have keys x and y. x contains n by 1000 rows. y contains n by 919 labels.
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

    # Running in TRAIN, VALID, or TEST?    
    mode = kwargs.get("mode")
    # specifies the indices to generate records.
    # can be used for debug purposes, or to evaluate
    # only specific regions in a vector
    # TODO AM 4/17/2019: move this to an actual parameter
    indices = kwargs.get("indices")
    
    if (not isinstance(indices, np.ndarray)):
        indices = range(0, data["y"].shape[-1]) # if not defined, set to all points
    
    if (not isinstance(mode, Dataset)):
        raise ValueError("mode is not a Dataset enum")
        
    if (mode == Dataset.RUNTIME):
        label_cell_types = ["PLACEHOLDER_CELL"]
        dnase_vector = kwargs.get("dnase_vector")
        random_cell = list(cellmap)[0] # placeholder to get label vector length
        
    print("using %s as labels for mode %s" % (label_cell_types, mode))
        
        
    if (mode == Dataset.TEST or mode == Dataset.RUNTIME):
        # Drop cell types with the least information (TODO AM 4/1/2019 this could do something smarter)
        
        # make dictionary of eval_cell_type: assay count and sort in decreasing order
        tmp = matrix.copy()
        tmp[tmp>= 0] = 1
        tmp[tmp== -1] = 0
        sums = np.sum(tmp, axis = 1)
        cell_assay_counts = zip(list(cellmap), sums)
        cell_assay_counts = sorted(cell_assay_counts, key = lambda x: x[1])
        # filter by eval_cell_types
        cell_assay_counts = list(filter(lambda x: x[0] in eval_cell_types, cell_assay_counts))
        
        # remove cell types with smallest number of factors
        eval_cell_types = eval_cell_types.copy()
        [eval_cell_types.remove(i[0]) for i in cell_assay_counts[0:len(label_cell_types)]]
        del tmp
        del cell_assay_counts
        
    def g():
                
        for i in indices: # for all records specified
            
            for (cell) in label_cell_types: # for all cell types to be used in labels
                dnases = [] 
                
                # cells to be featurized
                feature_cells = eval_cell_types.copy()
                
                # try to remove cell if it is in the possible list of feature cell types
                try:
                    feature_cells.remove(cell) 
                except ValueError:
                    pass  # do nothing!
                
                # features from all remaining cells not in label set
                feature_cell_indices_list = list(map(lambda c: get_y_indices_for_cell(matrix, cellmap, c), feature_cells))
                feature_cell_indices = np.array(feature_cell_indices_list).flatten()
                
                # labels for this cell
                if (mode != Dataset.RUNTIME):
                    label_cell_indices = get_y_indices_for_cell(matrix, cellmap, cell)
                    label_cell_indices_no_dnase = np.delete(label_cell_indices, [0])

                    # Copy assay_index_no_dnase and turn into mask of 0/1 for whether data for this cell type for
                    # a given label is available.
                    assay_mask = np.copy(label_cell_indices_no_dnase)
                    assay_mask[assay_mask == -1] = 0
                    assay_mask[assay_mask > 0] = 1
                    
                else:
                    label_count = len(get_y_indices_for_cell(matrix, cellmap, random_cell))-1
                    
                    # Mask and labels are all 0's because labels are missing during runtime
                    garbage_labels = assay_mask = np.zeros(label_count)

                # get dnase indices for cell types that are going to be features
                dnase_indices = [x[0] for x in feature_cell_indices_list]
            
                for radius in radii:
                    min_radius = max(0, i - radius)
                    max_radius = i+radius+1
                    
                    # use DNase vector, if it is provided
                    if (mode == Dataset.RUNTIME):

                        # within the radius, fraction of places where they are both 1
                        dnase_double_positive = np.average(data["y"][dnase_indices,min_radius:max_radius]*
                                                 dnase_vector[min_radius:max_radius], axis=1)

                        # within the radius, fraction of places where they are both equal (0 or 1)
                        dnase_agreement = np.average(data["y"][dnase_indices,min_radius:max_radius]==
                                                 dnase_vector[min_radius:max_radius], axis=1)

                    else:
                        # within the radius, fraction of places where they are both 1
                        # label_cell_index[0] == DNase location for specific cell type
                        dnase_double_positive = np.average(data["y"][dnase_indices,min_radius:max_radius]*
                                                 data["y"][label_cell_indices[0],min_radius:max_radius], axis=1)

                        # within the radius, fraction of places where they are both equal (0 or 1)
                        dnase_agreement = np.average(data["y"][dnase_indices,min_radius:max_radius]==
                                                 data["y"][label_cell_indices[0],min_radius:max_radius], axis=1)
                        
                        
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

                # There can be NaNs in the DNases for edge cases (when the radii extends past the end of the chr).
                # Mask these values in the first row of tmp
                tmp[0,np.where(np.isnan(tmp[1,:]))[0]] = 0 # assign UNKs to missing DNase values
                tmp[1,np.where(np.isnan(tmp[1,:]))[0]] = 0 # set NaNs to 0
                
                if (mode != Dataset.RUNTIME):
                    # yield features, labels, label mask
                    yield tmp, \
                             data["y"][label_cell_indices_no_dnase,i], \
                             assay_mask

                else:
                    yield tmp, garbage_labels, assay_mask

    return g


############################################################################################
######################## Functions for running data generators #############################
############################################################################################

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

    # only set output_shapes if g() has data
    try: 
        dataset = tf.data.Dataset.from_generator(
            g,
            output_types=(tf.float32,)*3, # 3 = features, labels, and missing indices
            output_shapes=(x.shape, y.shape, z.shape,)
        )
    except NameError as e:
        print("Error: no data, %s" % e)
        dataset = tf.data.Dataset.from_generator(
            g,
            output_types=(tf.float32,)*3 # 3 = features, labels, and missing indices
        )
        
    dataset = dataset.batch(batch_size)
    dataset = dataset.shuffle(shuffle_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(prefetch_size)

    try: 
        y
        return y.shape, dataset.make_one_shot_iterator()
    except NameError as e:
        return None, dataset.make_one_shot_iterator()
