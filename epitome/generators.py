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

    
############################### Channel generator ################################

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
            indices = range(0, data.shape[-1]) # not training mode, set to all points

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
        
    def g():
        for i in indices: # for all records specified
            for (cell) in label_cell_types: # for all cell types to be used in labels
                dnases = [] 
                dnases_double_positive = []
                dnases_agreement = []

                # features from all remaining cells not in label set
                feature_cell_indices_list = list(map(lambda c: get_y_indices_for_cell(matrix, cellmap, c), 
                                                     eval_cell_types))
                # remove missing indices
                feature_cell_indices_list = [m[m!=-1] for m in feature_cell_indices_list]
                
                # labels for this cell
                if (mode != Dataset.RUNTIME):
                    label_cell_indices = get_y_indices_for_cell(matrix, cellmap, cell)
                    label_cell_indices_no_dnase = np.delete(label_cell_indices, [0]) # delete dnase

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
                dnases_agreement_reshaped = np.array(dnases_agreement).reshape([len(radii), len(eval_cell_types)]).T
                dnases_double_positive_reshaped = np.array(dnases_double_positive).reshape([len(radii), len(eval_cell_types)]).T

                dnases = np.concatenate([dnases_agreement_reshaped, dnases_double_positive_reshaped], axis=1)

                final = []
                for j,c in enumerate(eval_cell_types):
                    cell_features = data[feature_cell_indices_list[j],i]
                    cell_dnase = dnases[j,:]
                    concat = np.concatenate([cell_features, cell_dnase])
                    if c == cell: # if eval cell write out missing values
                        final.append(np.zeros(len(concat)))
                    else:
                        final.append(concat)
                    
                
                if (mode != Dataset.RUNTIME):
                    labels = data[label_cell_indices_no_dnase,i]

                else: # used when just predicting
                    # The features going into the example.
                    labels = garbage_labels # all 0's

                # append labels and assaymask
                final.append(labels.astype(np.float32))
                final.append(assay_mask.astype(np.float32))
                yield tuple(final)

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
    
    for f in g():
        break
    labels = f[-2]
    assay_mask = f[-1]
    features = f[:-2]
    shapes = []
    
    for i in f:
        shapes.append(i.shape)

    try: 
        dataset = tf.data.Dataset.from_generator(
            g,
            output_types=(tf.float32,)* len(f), 
            output_shapes=tuple(shapes)
        )
    except NameError as e:
        print("Error: no data, %s" % e)
        dataset = tf.data.Dataset.from_generator(
            g,
            output_types=(tf.float32,)*len(features) 
        )

    dataset = dataset.batch(batch_size)
    dataset = dataset.shuffle(shuffle_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(prefetch_size)
    
    try: 
        features
        return [i.shape[0] for i in features], labels.shape, dataset
    except NameError as e:
        return None, None, dataset

    
    
