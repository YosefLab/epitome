"""
Functions for data generators.
"""

import numpy as np
import tensorflow as tf
from .constants import *
from .functions import *
import epitome.iio as iio
import glob

######################### Original Data Generator: Only peak based #####################
np.random.seed(0) # to keep np.random.choice consistent

def load_data(data, 
                 label_cell_types,  # used for labels. Should be all for train/eval and subset for test
                 eval_cell_types,   # used for rotating features. Should be all - test for train/eval
                 matrix,
                 assaymap,
                 cellmap,
                 radii,
                 **kwargs):
    
    # AM 5/20/2019. This is enforcing exclusive DNase bins and will make 
    # interpretation of DNase weights easier. It does not add performance benefit
    # over inclusive bins, which was used in the original model.
    exclusive = True

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
    
    if (not isinstance(indices, np.ndarray) and not isinstance(indices, list)):
        # model performs better when there are less 0s
        if mode == Dataset.TRAIN:
            feature_indices = np.concatenate(list(map(lambda c: get_y_indices_for_cell(matrix, cellmap, c), 
                                     list(cellmap))))
            feature_indices = feature_indices[feature_indices != -1]

            # need to re-proportion the indices to equalize positives
            if (len(list(assaymap)) > 2):

                # get sums for each feature in the dataset
                rowsums = np.sum(data[feature_indices,:], axis=1) 

                # multiply data by row scaling factor
                scale_factor = 1/rowsums
                scaled = data[feature_indices,:] * scale_factor[:, np.newaxis] 

                # indices where sum > 0
                indices_zero = np.where(np.sum(scaled, axis=0) > 0)[0]
                # then filter indices by probabilities inversely proportional to frequency
                indices = np.random.choice(indices_zero, int(indices_zero.shape[0] * 0.4), p=(np.sum(scaled, axis=0)/np.sum(scaled))[indices_zero])

            else:
                # single TF model
                feature_indices = np.concatenate(list(map(lambda c: get_y_indices_for_cell(matrix, cellmap, c), 
                                                     list(cellmap))))

                # chop of DNase
                TF_indices = feature_indices.reshape([len(cellmap),len(assaymap)])[:,1]
                
                TF_indices =  TF_indices[TF_indices != -1]
                feature_indices = feature_indices[feature_indices != -1]

                # sites where TF is in at least 1 cell line
                positive_indices = np.where(np.sum(data[TF_indices,:], axis=0) > 1)[0]

                indices_probs = np.ones([data.shape[1]])
                indices_probs[positive_indices] = 0
                indices_probs = indices_probs/np.sum(indices_probs, keepdims=1)

                # randomly select 10 fold sites where TF is not in any cell line
                negative_indices = np.random.choice(np.arange(0,data.shape[1]), positive_indices.shape[0] * 10,p=indices_probs)
                indices = np.sort(np.concatenate([negative_indices, positive_indices]))

                
        else:
            indices = range(0, data.shape[-1]) # if not defined, set to all points
    
    if (not isinstance(mode, Dataset)):
        raise ValueError("mode is not a Dataset enum")
        
    if (mode == Dataset.RUNTIME):
        label_cell_types = ["PLACEHOLDER_CELL"]
        dnase_vector = kwargs.get("dnase_vector")
        random_cell = list(cellmap)[0] # placeholder to get label vector length
        
    print("using %s as labels for mode %s" % (label_cell_types, mode))
    
    # string of radii for meta data labeling
    radii_str = list(map(lambda x: "DNASE_RADII_%i" % x, radii))
        
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
        eval_cell_types = list(eval_cell_types)
        [eval_cell_types.remove(i[0]) for i in cell_assay_counts[0:len(label_cell_types)]]
        del tmp
        del cell_assay_counts
        
    def g():
        for i in indices: # for all records specified
            for (cell) in label_cell_types: # for all cell types to be used in labels
                dnases = [] 
                dnases_double_positive = []
                dnases_agreement = []
                
                # cells to be featurized
                feature_cells = list(eval_cell_types)
                
                # try to remove cell if it is in the possible list of feature cell types
                try:
                    feature_cells.remove(cell) 
                except ValueError:
                    pass  # do nothing!
                                
                # features from all remaining cells not in label set
                feature_cell_indices_list = list(map(lambda c: get_y_indices_for_cell(matrix, cellmap, c), 
                                                     feature_cells))
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
                dnase_indices = np.array([x[0] for x in feature_cell_indices_list])
                
                for r, radius in enumerate(radii):
                    
                    min_radius = max(0, i - radius + 1)
                    max_radius = min(i+radius, data.shape[1])
                    
                    # if exclusive == True, then do not featurize chromatin regions
                    # that were considered in smaller radii
                    if (exclusive and r != 0):
                        radius_range_1 = np.arange(min_radius, max(0, i - radii[r-1]+1))
                        radius_range_2 = np.arange(i+radii[r-1], max_radius)
                        
                        radius_range = np.concatenate([radius_range_1, radius_range_2])
                    else:
                        
                        radius_range = np.arange(min_radius, max_radius)
                        
                        
                    ####################################################################
                    
                    # use DNase vector, if it is provided
                    if (mode == Dataset.RUNTIME):

                        # within the radius, fraction of places where they are both 1
                        dnase_double_positive = np.average(data[dnase_indices[:,None],radius_range]*
                                                 dnase_vector[radius_range], axis=1)

                        # within the radius, fraction of places where they are both equal (0 or 1)
                        dnase_agreement = np.average(data[dnase_indices[:,None],radius_range]==
                                                 dnase_vector[radius_range], axis=1)

                    else:
                        # within the radius, fraction of places where they are both 1
                        # label_cell_index[0] == DNase location for specific cell type
                        dnase_double_positive = np.average(data[dnase_indices[:,None],radius_range]*
                                                 data[label_cell_indices[0],radius_range], axis=1)

                        # within the radius, fraction of places where they are both equal (0 or 1)
                        dnase_agreement = np.average(data[dnase_indices[:,None],radius_range]==
                                                 data[label_cell_indices[0],radius_range], axis=1)
                        
                    dnases_double_positive.extend(dnase_double_positive)
                    dnases_agreement.extend(dnase_agreement)
                        
                # rehape agreement DNase to Radii by feature_cells
                dnases_agreement_reshaped = np.array(dnases_agreement).reshape([len(radii), len(feature_cells)])
                dnases_double_positive_reshaped = np.array(dnases_double_positive).reshape([len(radii), len(feature_cells)])
                dnase_means = np.mean(dnases_agreement_reshaped, axis = 0)

                ######### reorder cells by similarity ################
                ## This was added 5/30/2019. It seems to *maybe help 
                ## a little bit on cell types not seen in the model.
                ## This makes sense because cell types are now ordered
                ## by similarity and keep some spacial positioning 
                ## based on the similarity to the new cell. 
                best_indices = (-dnase_means).argsort()

                dnases.extend(dnases_double_positive_reshaped[:,best_indices].flatten())
                dnases.extend(dnases_agreement_reshaped[:,best_indices].flatten())

                feature_cell_indices = feature_cell_indices.reshape([len(feature_cells), len(assaymap)])[best_indices,:].flatten()
                ######## End reorder #################################                                                   
                                                                        
                                                                        
                # Extract features
                features = data[feature_cell_indices,i]
                
                # concatenate features and DNases
                x_data = np.concatenate([features, dnases])
                
                # mask for x_data. 0 = do not mask, 1 = mask.
                x_mask = np.zeros(x_data.shape[0])
                x_mask[np.where(feature_cell_indices == -1)[0]] = True # assign mask to missing features

                # There can be NaNs in the DNases for edge cases (when the radii extends past the end of the chr).
                # Mask these values in the first row of tmp
                x_mask[np.where(np.isnan(x_data))[0]] = True # assign mask to missing DNase values
                x_data[np.where(x_mask == True)[0]] = 0 # set all UNKs to 0
                
                # mask x data by which factors are available for a cell type
                x_data_masked = np.vstack([x_mask, x_data]) # top row 0 = mask, bottom row 1 = data
                        
                if (mode != Dataset.RUNTIME):
                    labels = data[label_cell_indices_no_dnase,i]

                else: # used when just predicting
                    # The features going into the example.
                    labels = garbage_labels # all 0's
                    
                yield (x_data_masked, labels, assay_mask)

    return g



def load_v_data(data, 
                 label_cell_types,  # used for labels. Should be all for train/eval and subset for test
                 eval_cell_types,   # used for rotating features. Should be all - test for train/eval
                 matrix,
                 assaymap,
                 cellmap,
                 radii,
                 **kwargs):
    
    # AM 5/20/2019. This is enforcing exclusive DNase bins and will make 
    # interpretation of DNase weights easier. It does not add performance benefit
    # over inclusive bins, which was used in the original model.
    exclusive = True

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
    # set all UNKs features to TF mean
    def TF_mean(i, idx, cell_indices):
        """
        Gets the mean TF binding score for an element in this location
        """
        # compute means at row i in data for all assays
        indices = matrix[cell_indices,:].T.flatten()
        tmp1 = data[:,i][indices].astype(None)
        tmp1[tmp1 == Label.UNK.value] = np.nan
        tmp1[tmp1 == Label.IMPUTED_BOUND.value] = Label.BOUND.value
        tmp1[tmp1 == Label.IMPUTED_UNBOUND.value] = Label.UNBOUND.value
        means = np.nanmean(tmp1.reshape([len(cell_indices),  matrix.shape[1]]), axis=0) 
        
        TF_idx = idx % matrix.shape[1]
        return means[TF_idx]

    # Running in TRAIN, VALID, or TEST?    
    mode = kwargs.get("mode")
    # specifies the indices to generate records.
    # can be used for debug purposes, or to evaluate
    # only specific regions in a vector
    # TODO AM 4/17/2019: move this to an actual parameter
    indices = kwargs.get("indices")
    
    if (not isinstance(indices, np.ndarray) and not isinstance(indices, list)):
        indices = range(0, data.shape[-1]) # if not defined, set to all points
    
    if (not isinstance(mode, Dataset)):
        raise ValueError("mode is not a Dataset enum")
        
    if (mode == Dataset.RUNTIME):
        label_cell_types = ["PLACEHOLDER_CELL"]
        dnase_vector = kwargs.get("dnase_vector")
        random_cell = list(cellmap)[0] # placeholder to get label vector length
        
    print("using %s as labels for mode %s" % (label_cell_types, mode))
    
    # string of radii for meta data labeling
    radii_str = list(map(lambda x: "DNASE_RADII_%i" % x, radii))
        
    if (mode == Dataset.TEST or mode == Dataset.RUNTIME):
        # Drop cell types with the least information (TODO AM 4/1/2019 this could do something smarter)
        
        # make dictionary of eval_cell_type: assay count and sort in decreasing order
        tmp = matrix.copy()
        tmp[tmp >= Label.UNBOUND.value] = 1
        tmp[tmp == Label.UNK.value] = 0 
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
                dnases_double_positive = []
                dnases_agreement = []
                
                # cells to be featurized
                feature_cells = eval_cell_types.copy()
                
                # try to remove cell if it is in the possible list of feature cell types
                try:
                    feature_cells.remove(cell) 
                except ValueError:
                    pass  # do nothing!
                                
                # features from all remaining cells not in label set
                feature_cell_indices_list = list(map(lambda c: get_y_indices_for_cell(matrix, cellmap, c), 
                                                     feature_cells))
                feature_cell_indices = np.array(feature_cell_indices_list).flatten()
                
                                
                # labels for this cell
                if (mode != Dataset.RUNTIME):
                    label_cell_indices = get_y_indices_for_cell(matrix, cellmap, cell)
                    label_cell_indices_no_dnase = np.delete(label_cell_indices, [0])
                    labels = data[label_cell_indices_no_dnase,i]

                    # Copy assay_index_no_dnase and turn into mask of 0/1 for whether data for this cell type for
                    # a given label is available.
                    assay_mask = np.ones(labels.shape)
                    # set mask over missing AND imputed mask
                    assay_mask[labels < Label.UNBOUND.value] = 0
                else:
                    label_count = len(get_y_indices_for_cell(matrix, cellmap, random_cell))-1
                    
                    # Mask and labels are all 0's because labels are missing during runtime
                    garbage_labels = assay_mask = np.zeros(label_count)
                    # The features going into the example.
                    labels = garbage_labels # all 0's

                # get dnase indices for cell types that are going to be features
                dnase_indices = np.array([x[0] for x in feature_cell_indices_list])
                
                for r, radius in enumerate(radii):
                    
                    min_radius = max(0, i - radius + 1)
                    max_radius = min(i+radius, data.shape[1])
                    
                    # if exclusive == True, then do not featurize chromatin regions
                    # that were considered in smaller radii
                    if (exclusive and r != 0):
                        radius_range_1 = np.arange(min_radius, max(0, i - radii[r-1]+1))
                        radius_range_2 = np.arange(i+radii[r-1], max_radius)
                        
                        radius_range = np.concatenate([radius_range_1, radius_range_2])
                    else:
                        
                        radius_range = np.arange(min_radius, max_radius)
                        
                        
                    ####################################################################
                    
                    # use DNase vector, if it is provided
                    if (mode == Dataset.RUNTIME):

                        # within the radius, fraction of places where they are both 1
                        dnase_double_positive = np.average(data[dnase_indices[:,None],radius_range]*
                                                 dnase_vector[radius_range], axis=1)

                        # within the radius, fraction of places where they are both equal (0 or 1)
                        dnase_agreement = np.average(data[dnase_indices[:,None],radius_range]==
                                                 dnase_vector[radius_range], axis=1)

                    else:
                        # within the radius, fraction of places where they are both 1
                        # label_cell_index[0] == DNase location for specific cell type
                        dnase_double_positive = np.average(data[dnase_indices[:,None],radius_range]*
                                                 data[label_cell_indices[0],radius_range], axis=1)

                        # within the radius, fraction of places where they are both equal (0 or 1)
                        dnase_agreement = np.average(data[dnase_indices[:,None],radius_range]==
                                                 data[label_cell_indices[0],radius_range], axis=1)
                        
                    dnases_double_positive.extend(dnase_double_positive)
                    dnases_agreement.extend(dnase_agreement)
                        
                # rehape agreement DNase to Radii by feature_cells
                dnases_agreement_reshaped = np.array(dnases_agreement).reshape([len(radii), len(feature_cells)])
                dnases_double_positive_reshaped = np.array(dnases_double_positive).reshape([len(radii), len(feature_cells)])
                dnase_means = np.mean(dnases_agreement_reshaped, axis = 0)

                ######### reorder cells by similarity ################
                ## This was added 5/30/2019. It seems to *maybe help 
                ## a little bit on cell types not seen in the model.
                ## This makes sense because cell types are now ordered
                ## by similarity and keep some spacial positioning 
                ## based on the similarity to the new cell. 
                best_indices = (-dnase_means).argsort()

                dnases.extend(dnases_double_positive_reshaped[:,best_indices].flatten())
                dnases.extend(dnases_agreement_reshaped[:,best_indices].flatten())

                feature_cell_indices = feature_cell_indices.reshape([len(feature_cells), len(assaymap)])[best_indices,:].flatten()
                ######## End reorder #################################                                                   
                                                                        
                                                                        
                # Extract features
                features = data[feature_cell_indices,i]
                
                # concatenate features and DNases
                x_data = np.concatenate([features, dnases])
                
                # mask for x_data. 0 = do not mask, 1 = mask.
                x_mask = np.zeros(x_data.shape[0])
                # mask imputed (bound and unbound) and missing features
                x_mask[np.where(features == Label.UNK.value)[0]] = True # assign mask to missing features

                # There can be NaNs in the DNases for edge cases (when the radii extends past the end of the chr).
                # Mask these values in the first row of tmp
                x_mask[np.where(np.isnan(x_data))[0]] = True # assign mask to missing DNase values
                x_data[np.where(x_mask == True)[0]] = 0 # set all UNKs to 0
                
                # reset imputed values to normal known values in x_data
                x_data[np.where(x_data == Label.IMPUTED_BOUND.value)[0]] = Label.BOUND.value 
                x_data[np.where(x_data == Label.IMPUTED_UNBOUND.value)[0]] = Label.UNBOUND.value 
                
                # recalculate missing values as means
                masked_indices = np.where(x_mask == True)[0][:len(feature_cell_indices)]
                cellmap_idx = list(map(lambda x: cellmap[x], feature_cells))
                x_data[masked_indices] = TF_mean(i, masked_indices, cellmap_idx)
                
                yield (x_data, labels, assay_mask)

    return g

def shuffled_data(data, 
                 label_cell_types,  # used for labels. Should be all for train/eval and subset for test
                 eval_cell_types,   # used for rotating features. Should be all - test for train/eval
                 matrix,
                 assaymap,
                 cellmap,
                 radii,
                 **kwargs):
    
    # AM 5/20/2019. This is enforcing exclusive DNase bins and will make 
    # interpretation of DNase weights easier. It does not add performance benefit
    # over inclusive bins, which was used in the original model.
    exclusive = True

    """
    Shuffles features. Used to test uncertainty in model.
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
    
    if (not isinstance(indices, np.ndarray) and not isinstance(indices, list)):
        indices = range(0, data.shape[-1]) # if not defined, set to all points
    
    if (not isinstance(mode, Dataset)):
        raise ValueError("mode is not a Dataset enum")
        
    if (mode == Dataset.RUNTIME):
        label_cell_types = ["PLACEHOLDER_CELL"]
        dnase_vector = kwargs.get("dnase_vector")
        random_cell = list(cellmap)[0] # placeholder to get label vector length
        
    print("using %s as labels for mode %s" % (label_cell_types, mode))
    
    # string of radii for meta data labeling
    radii_str = list(map(lambda x: "DNASE_RADII_%i" % x, radii))
        
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
        eval_cell_types = list(eval_cell_types)
        [eval_cell_types.remove(i[0]) for i in cell_assay_counts[0:len(label_cell_types)]]
        del tmp
        del cell_assay_counts
        
        
    def g():
        for i in indices: # for all records specified
            for (cell) in label_cell_types: # for all cell types to be used in labels
                dnases = [] 
                dnases_double_positive = []
                dnases_agreement = []
                
                # cells to be featurized
                feature_cells = list(eval_cell_types)
                
                # try to remove cell if it is in the possible list of feature cell types
                try:
                    feature_cells.remove(cell) 
                except ValueError:
                    pass  # do nothing!
                                
                # features from all remaining cells not in label set
                feature_cell_indices_list = list(map(lambda c: get_y_indices_for_cell(matrix, cellmap, c), 
                                                     feature_cells))
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
                dnase_indices = np.array([x[0] for x in feature_cell_indices_list])
                
                for r, radius in enumerate(radii):
                    
                    min_radius = max(0, i - radius + 1)
                    max_radius = min(i+radius, data.shape[1])
                    
                    # if exclusive == True, then do not featurize chromatin regions
                    # that were considered in smaller radii
                    if (exclusive and r != 0):
                        radius_range_1 = np.arange(min_radius, max(0, i - radii[r-1]+1))
                        radius_range_2 = np.arange(i+radii[r-1], max_radius)
                        
                        radius_range = np.concatenate([radius_range_1, radius_range_2])
                    else:
                        
                        radius_range = np.arange(min_radius, max_radius)
                        
                        
                    ####################################################################
                    
                    # use DNase vector, if it is provided
                    if (mode == Dataset.RUNTIME):

                        # within the radius, fraction of places where they are both 1
                        dnase_double_positive = np.average(data[dnase_indices[:,None],radius_range]*
                                                 dnase_vector[radius_range], axis=1)

                        # within the radius, fraction of places where they are both equal (0 or 1)
                        dnase_agreement = np.average(data[dnase_indices[:,None],radius_range]==
                                                 dnase_vector[radius_range], axis=1)

                    else:
                        # within the radius, fraction of places where they are both 1
                        # label_cell_index[0] == DNase location for specific cell type
                        dnase_double_positive = np.average(data[dnase_indices[:,None],radius_range]*
                                                 data[label_cell_indices[0],radius_range], axis=1)

                        # within the radius, fraction of places where they are both equal (0 or 1)
                        dnase_agreement = np.average(data[dnase_indices[:,None],radius_range]==
                                                 data[label_cell_indices[0],radius_range], axis=1)
                        
                    dnases_double_positive.extend(dnase_double_positive)
                    dnases_agreement.extend(dnase_agreement)
                        
                # rehape agreement DNase to Radii by feature_cells
                dnases_agreement_reshaped = np.array(dnases_agreement).reshape([len(radii), len(feature_cells)])
                dnases_double_positive_reshaped = np.array(dnases_double_positive).reshape([len(radii), len(feature_cells)])
                dnase_means = np.mean(dnases_agreement_reshaped, axis = 0)

                ######### reorder cells by similarity ################
                ## This was added 5/30/2019. It seems to *maybe help 
                ## a little bit on cell types not seen in the model.
                ## This makes sense because cell types are now ordered
                ## by similarity and keep some spacial positioning 
                ## based on the similarity to the new cell. 
                best_indices = (-dnase_means).argsort()

                dnases.extend(dnases_double_positive_reshaped[:,best_indices].flatten())
                dnases.extend(dnases_agreement_reshaped[:,best_indices].flatten())

                feature_cell_indices = feature_cell_indices.reshape([len(feature_cells), len(assaymap)])[best_indices,:].flatten()
                ######## End reorder #################################                                                   
                                                                        
                                                                        
                # Extract features
                features = data[feature_cell_indices,i]
                
                # concatenate features and DNases
                x_data = np.concatenate([features, dnases])
                
                # mask for x_data. 0 = do not mask, 1 = mask.
                x_mask = np.zeros(x_data.shape[0])
                x_mask[np.where(feature_cell_indices == -1)[0]] = True # assign mask to missing features

                # There can be NaNs in the DNases for edge cases (when the radii extends past the end of the chr).
                # Mask these values in the first row of tmp
                x_mask[np.where(np.isnan(x_data))[0]] = True # assign mask to missing DNase values
                x_data[np.where(x_mask == True)[0]] = 0 # set all UNKs to 0
                
                # mask x data by which factors are available for a cell type
                x_data_masked = np.vstack([x_mask, x_data]) # top row 0 = mask, bottom row 1 = data
                np.random.shuffle(x_data_masked )
                        
                if (mode != Dataset.RUNTIME):
                    labels = data[label_cell_indices_no_dnase,i]

                else: # used when just predicting
                    # The features going into the example.
                    labels = garbage_labels # all 0's
                    
                yield (x_data_masked, labels, assay_mask)

    return g


def generator_to_tf_dataset(g, batch_size, shuffle_size, prefetch_size):
    """
    Generates a tensorflow dataset from a data generator.
    
    :param g: data generator
    :param batch_size: number of elements in generator to combine into a single batch
    :param shuffle_size: number of elements from the  generator fromw which the new dataset will shuffle
    :param prefetch_size: maximum number of elements that will be buffered  when prefetching
    
    :returns: tuple of (label shape, tf.data.Dataset)
    """
    
    for x, y, y_mask in g():
        break
    
    try: 
        dataset = tf.data.Dataset.from_generator(
            g,
            output_types=(tf.float32,)*3, # 3 = features, labels, and missing indices
            output_shapes=(x.shape, y.shape, y.shape,)
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
        return y.shape, dataset
    except NameError as e:
        return None, dataset
