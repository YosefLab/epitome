"""
Functions for data generators.
"""

import numpy as np
import tensorflow as tf
from .constants import *
from .functions import *
from .sampling import *
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
                 similarity_assays = ['DNase'],
                 mode = Dataset.TRAIN,
                 similarity_matrix = None,
                 indices = None,
                 return_feature_names = False,
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
    :param radii: radii to compute similarity distances from
    :param mode: Dataset.TRAIN, VALID, TEST or RUNTIME
    :param indices: indices in genome to generate records for.
    :param kwargs: kargs

    :returns: generator of data with three elements:
        1. record features
        2. record labels for a given cell type
        3. 0/1 mask of labels that have validation data. For example, if this record is for celltype A549,
        and A549 does not have data for ATF3, there will be a 0 in the position corresponding to the label space.
    """

    # for now, we require DNase to be part of the similarity comparison
    assert('DNase' in similarity_assays)
        
    # get indices for features.rows are cells and cols are assays
    cellmap_idx = [cellmap[c] for c in list(eval_cell_types)]
    feature_cell_indices = matrix[cellmap_idx,:]
    
    # indices to be deleted used for similarity comparison
    delete_indices = np.array([assaymap[s] for s in similarity_assays])
    
    # make sure no similarity comparison data is missing for all cell types
    assert np.invert(np.any(feature_cell_indices[:,delete_indices] == -1)), \
        "missing data at %s" % (np.where(feature_cell_indices[:,delete_indices] == -1)[0])
    
    # names of labels that are being predicted
    feature_assays = [a for a in list(assaymap)] # assays used as features for each evaluation cell type
    label_assays = [a for a in feature_assays if a not in similarity_assays]

    if (not isinstance(mode, Dataset)):
        raise ValueError("mode is not a Dataset enum")

    if (not isinstance(indices, np.ndarray) and not isinstance(indices, list)):
        # model performs better when there are less 0s
        if mode == Dataset.TRAIN:
            feature_indices = np.concatenate(list(map(lambda c: get_y_indices_for_cell(matrix, cellmap, c),
                                     list(cellmap))))
            feature_indices = feature_indices[feature_indices != -1]

            # need to re-proportion the indices to oversample underrepresented labels
            if (len(list(assaymap)) > 2):
                # configure y: label matrix of ChIP for all assays from all cell lines in train
                indices = np.concatenate([get_y_indices_for_assay(matrix, assaymap, assay) for assay in label_assays])
                indices = indices[indices != -1]
                y = data[indices, :].T
                m = MLSMOTE(y)
                indices = m.fit_resample()
            
            else:
                # single TF model
                # get indices for DNAse and chip for this mark
                feature_indices = np.concatenate(list(map(lambda c: get_y_indices_for_cell(matrix, cellmap, c),
                                                     list(cellmap))))

                # chop off assays being used in similarity metric
                not_similarity_indices = np.array([v for k,v in assaymap.items() if k not in similarity_assays])
                TF_indices = feature_indices.reshape([len(cellmap),len(assaymap)])[:,not_similarity_indices]

                TF_indices =  TF_indices[TF_indices != -1]
                feature_indices = feature_indices[feature_indices != -1]

                # sites where TF is bound in at least 2 cell line
                positive_indices = np.where(np.sum(data[TF_indices,:], axis=0) > 1)[0]

                indices_probs = np.ones([data.shape[1]])
                indices_probs[positive_indices] = 0
                indices_probs = indices_probs/np.sum(indices_probs, keepdims=1)

                # randomly select 10 fold sites where TF is not in any cell line
                negative_indices = np.random.choice(np.arange(0,data.shape[1]), 
                                                    positive_indices.shape[0] * 10,
                                                    p=indices_probs)
                indices = np.sort(np.concatenate([negative_indices, positive_indices]))

        else:
            indices = range(0, data.shape[-1]) # not training mode, set to all points


    if (mode == Dataset.RUNTIME):
        label_cell_types = ["PLACEHOLDER_CELL"]
        if similarity_matrix is None:
            raise Exception("similarity_matrix must be defined in runtime mode")
        assert similarity_matrix.shape[0] == len(similarity_assays), \
            "similarity_matrix is missing data for assays (should have %i rows)" % (len(similarity_assays))
        random_cell = list(cellmap)[0] # placeholder to get label vector length

    print("using %s as labels for mode %s" % (label_cell_types, mode))

    # string of radii for meta data labeling
    radii_str = list(map(lambda x: "RADII_%i" % x, radii))
    
    def g():
        for i in indices: # for all records specified
            feature_names = []
            
            for (cell) in label_cell_types: # for all cell types to be used in labels
                similarities_double_positive = np.empty([len(eval_cell_types),0])
                similarities_agreement = np.empty([len(eval_cell_types),0])

                # labels for this cell
                if (mode != Dataset.RUNTIME):
                    label_cell_indices = get_y_indices_for_cell(matrix, cellmap, cell)

                    # delete all indices being used in the similarity computation
                    label_cell_indices_no_similarities = np.delete(label_cell_indices, delete_indices)

                    # Copy assay_index_no_similarities and turn into mask of 0/1 for whether data for this cell type for
                    # a given label is available.
                    assay_mask = np.copy(label_cell_indices_no_similarities)
                    assay_mask[assay_mask == -1] = 0
                    assay_mask[assay_mask > 0] = 1

                else:
                    label_count = len(get_y_indices_for_cell(matrix, cellmap, random_cell))-len(similarity_assays)

                    # Mask and labels are all 0's because labels are missing during runtime
                    garbage_labels = assay_mask = np.zeros(label_count)
                

                # get indices for assays used in similarity computation
                # for cell types that are going to be features
                similarity_indices = feature_cell_indices[:, delete_indices]
                
                similarity_labels_agreement = []
                similarity_labels_dp = []
                
                for r, radius in enumerate(radii):

                    min_radius = max(0, i - radius + 1)
                    max_radius = min(i+radius, data.shape[1])

                    # do not featurize chromatin regions
                    # that were considered in smaller radii
                    if (r != 0):
                        radius_range_1 = np.arange(min_radius, max(0, i - radii[r-1]+1))
                        radius_range_2 = np.arange(i+radii[r-1], max_radius)

                        radius_range = np.concatenate([radius_range_1, radius_range_2])
                    else:

                        radius_range = np.arange(min_radius, max_radius)


                    ####################################################################
                    cell_train_data = data[similarity_indices[:,:,None],radius_range]

                    # use similarity matrix, if it is provided
                    if (mode == Dataset.RUNTIME):

                        # within the radius, fraction of places where they are both 1
                        similarity_double_positive = np.average(cell_train_data*
                                                 similarity_matrix[:,radius_range], axis=-1)

                        # within the radius, fraction of places where they are both equal (0 or 1)
                        similarity_agreement = np.average(cell_train_data==
                                                 similarity_matrix[:,radius_range], axis=-1)

                    else:
                        cell_label_data = data[label_cell_indices[delete_indices][:,None],radius_range]

                        similarity_double_positive = np.average(cell_train_data*
                                                 cell_label_data, axis=-1)

                        # within the radius, fraction of places where they are both equal (0 or 1)
                        similarity_agreement = np.average(cell_train_data ==
                                                 cell_label_data, axis=-1)

                    similarity_labels_agreement.append('r%i_%s' % (radius, 'agree'))
                    similarity_labels_dp.append('r%i_%s' % (radius, 'dp'))
                    
                    similarities_double_positive = np.concatenate([similarities_double_positive,similarity_double_positive],axis=1)
                    similarities_agreement = np.concatenate([similarities_agreement,similarity_agreement],axis=1)
                    
                # rehape agreement assay similarity to Radii by feature_cells
                similarities = np.concatenate([similarities_agreement, similarities_double_positive], axis=1)
                similarity_labels = np.concatenate([similarity_labels_agreement, similarity_labels_dp])

                final = []
                for j,c in enumerate(eval_cell_types):
                    # get indices for this cell that has data
                    present_indices = feature_cell_indices[j,:]
                    present_indices = present_indices[present_indices!=-1]
                    
                    cell_features = data[present_indices,i]
                    cell_similarities = similarities[j,:]
                    concat = np.concatenate([cell_features, cell_similarities])
                    final.append(concat)
                        
                    # concatenate together feature names
                    tmp = np.array(feature_assays)[feature_cell_indices[j,:] != -1]
                    al = ['%s_%s' % (c, a) for a in tmp]
                    sl = ['%s_%s' % (c, s) for s in similarity_labels]
                        
                    feature_names.append(np.concatenate([al, sl]))


                if (mode != Dataset.RUNTIME):
                    labels = data[label_cell_indices_no_similarities,i]

                else: # used when just predicting
                    # The features going into the example.
                    labels = garbage_labels # all 0's

                # append labels and assaymask
                final.append(labels.astype(np.float32))
                feature_names.append(['lbl_%s_%s' % (cell, a) for a in label_assays]) # of form lbl_cellline_target
                
                final.append(assay_mask.astype(np.float32))
                feature_names.append(['mask_%s_%s' % (cell, a) for a in label_assays]) # of form mask_cellline_target
                
                if (return_feature_names):
                    yield (tuple(final), tuple(feature_names))
                else:
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
