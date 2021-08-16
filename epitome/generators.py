r"""
========================
Data Generator Functions
========================
.. currentmodule:: epitome.generators

.. autosummary::
  :toctree: _generate/

  load_data
  generator_to_tf_dataset
"""

import numpy as np
import tensorflow as tf
from .constants import Dataset
from .functions import *
from .sampling import *
from .dataset import EpitomeDataset
import glob

######################### Original Data Generator: Only peak based #####################
np.random.seed(0) # to keep np.random.choice consistent


############################### Channel generator ################################
def load_data(data,
                 label_cell_types,  # used for labels. Should be all for train/eval and subset for test
                 eval_cell_types,   # used for rotating features. Should be all - test for train/eval
                 matrix,
                 targetmap,
                 cellmap,
                 radii,
                 similarity_targets = ['DNase'],
                 mode = Dataset.TRAIN,
                 similarity_matrix = None,
                 indices = None,
                 continuous = False,
                 return_feature_names = False,
                 **kwargs):
    """
    Takes Deepsea data and calculates distance metrics from cell types whose locations
    are specified by label_cell_indices, and the other cell types in the set. Label space is only one cell type.
    :param data: dictionary of matrices. Should have keys x and y. x contains n by 1000 rows. y contains n by 919 labels.
    :param label_cell_types: list of cell types to be rotated through and used as labels (subset of eval_cell_types)
    :param eval_cell_types: list of cell types to be used in evaluation (includes label_cell_types)
    :param matrix: matrix of celltype, target positions
    :param targetmap: map of column target positions in matrix
    :param cellmap: map of row cell type positions in matrix
    :param radii: radii to compute similarity distances from
    :param similarity_targets: list of targets used to measure cell type similarity (default is DNase-seq)
    :param mode: Dataset.TRAIN, VALID, TEST or RUNTIME
    :param similarity_matrix: matrix with shape (len(similarity_targets), genome_size) containing binary 0/1s of peaks for similarity_targets
    to be compared in the CASV.
    :param indices: indices in genome to generate records for.
    :param boolean continuous: determines whether similarity_matrix has continuous values. If continuous, we do not calculate agreement in the decreasing_train_valid_iters
      TODO: remove this eventually, if you can show agreement does not help performance
    :param return_feature_names: boolean whether to return string names of features
    :param kwargs: kargs

    :returns: generator of data with three elements:
        1. record features
        2. record labels for a given cell type
        3. 0/1 mask of labels that have validation data. For example, if this record is for celltype A549,
        and A549 does not have data for ATF3, there will be a 0 in the position corresponding to the label space.
    """

    # reshape similarity_matrix to a matrix if there is only one target
    if similarity_matrix is not None:
        if len(similarity_matrix.shape) == 1:
            similarity_matrix = similarity_matrix[None,:]

    if type(similarity_targets) is not list:
        similarity_targets = [similarity_targets]

    if len(similarity_targets) == 0 and len(radii) > 0:
        raise ValueError("Cannot set radii to anything if there are no similarity assays, but found len(radii)=%i" % len(radii))

    # get indices for features. rows are cells and cols are targets
    cellmap_idx = [cellmap[c] for c in list(eval_cell_types)]
    feature_cell_indices = matrix[cellmap_idx,:]

    # indices to be deleted. used for similarity comparison, not predictions.
    delete_indices = np.array([targetmap[s] for s in similarity_targets]).astype(int)

    # make sure no similarity comparison data is missing for all cell types
    assert np.invert(np.any(feature_cell_indices[:,delete_indices] == -1)), \
        "missing data for similarity target at %s" % (np.where(feature_cell_indices[:,delete_indices] == -1)[0])

    # names of labels that are being predicted
    feature_targets = [a for a in list(targetmap)] # targets used as features for each evaluation cell type
    label_targets = [a for a in feature_targets if a not in similarity_targets]

    if (not isinstance(mode, Dataset)):
        raise ValueError("mode is not a Dataset enum")

    if (not isinstance(indices, np.ndarray) and not isinstance(indices, list)):
        # model performs better when there are less 0s
        if mode == Dataset.TRAIN:
            feature_indices = np.concatenate(list(map(lambda c: EpitomeDataset.get_y_indices_for_cell(matrix, cellmap, c),
                                     list(cellmap))))
            feature_indices = feature_indices[feature_indices != -1]

            # need to re-proportion the indices to oversample underrepresented labels
            if (len(list(targetmap)) > 2):
                # configure y: label matrix of ChIP for all targets from all cell lines in train
                indices = np.concatenate([EpitomeDataset.get_y_indices_for_target(matrix, targetmap, target) for target in label_targets])
                indices = indices[indices != -1]
                y = data[indices, :].T
                m = MLSMOTE(y)
                indices = m.fit_resample()

            else:
                # single TF model
                # get indices for DNAse and chip for this mark
                feature_indices = np.concatenate(list(map(lambda c: EpitomeDataset.get_y_indices_for_cell(matrix, cellmap, c),
                                                     list(cellmap))))

                # chop off targets being used in similarity metric
                not_similarity_indices = np.array([v for k,v in targetmap.items() if k not in similarity_targets])
                TF_indices = feature_indices.reshape([len(cellmap),len(targetmap)])[:,not_similarity_indices]

                TF_indices =  TF_indices[TF_indices != -1]
                feature_indices = feature_indices[feature_indices != -1]

                # sites where TF is bound in at least 2 cell line
                positive_indices = np.where(np.sum(data[TF_indices,:], axis=0) > 1)[0]
                indices_probs = np.ones([data.shape[1]])
                indices_probs[positive_indices] = 0
                indices_probs = indices_probs/np.sum(indices_probs, keepdims=1)

                # If there are nans, it means there were no 0 cases.
                # We use this for testing so models converge quickly
                # with all ones.
                if np.any(np.isnan(indices_probs)):
                  print("Warning: no negative examples in dataset!!!")
                  indices_probs[:] = 1/indices_probs.shape[0]

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
        assert similarity_matrix.shape[0] == len(similarity_targets), \
            "similarity_matrix is missing data for targets (should have %i rows)" % (len(similarity_targets))
        random_cell = list(cellmap)[0] # placeholder to get label vector length

    print("using %s as labels for mode %s" % (label_cell_types, mode))

    # string of radii for meta data labeling
    radii_str = list(map(lambda x: "RADII_%i" % x, radii))

    def g():
        for i in indices: # for all records specified

            for (cell) in label_cell_types: # for all cell types to be used in labels

                # labels for this cell
                if (mode != Dataset.RUNTIME):
                    label_cell_indices = EpitomeDataset.get_y_indices_for_cell(matrix, cellmap, cell)

                    # delete all indices being used in the similarity computation
                    label_cell_indices_no_similarities = np.delete(label_cell_indices, delete_indices)

                    # Copy target_index_no_similarities and turn into mask of 0/1 for whether data for this cell type for
                    # a given label is available.
                    target_mask = np.copy(label_cell_indices_no_similarities)
                    target_mask[target_mask > -1] = 1
                    target_mask[target_mask == -1] = 0

                else:
                    label_count = len(EpitomeDataset.get_y_indices_for_cell(matrix, cellmap, random_cell))-len(similarity_targets)

                    # Mask and labels are all 0's because labels are missing during runtime
                    garbage_labels = target_mask = np.zeros(label_count)


                # get indices for targets used in similarity computation
                # for cell types that are going to be features
                similarity_indices = feature_cell_indices[:, delete_indices]


                # get indices for each radius in radii
                radius_ranges = list(map(lambda x: get_radius_indices(radii, x, i, data.shape[-1]), range(len(radii))))

                if len(radius_ranges) > 0:
                    radius_indices = np.concatenate(radius_ranges)

                    cell_train_data = data[similarity_indices[:,:,None],radius_indices]

                    if mode == Dataset.RUNTIME:

                        pos = cell_train_data*similarity_matrix[:,radius_indices]
                        agree = cell_train_data == similarity_matrix[:,radius_indices]

                    else:
                        cell_label_data = data[label_cell_indices[delete_indices][:,None],radius_indices]

                        # remove middle dimension and flatten similarity targets
                        pos = (cell_train_data*cell_label_data)
                        agree = (cell_train_data == cell_label_data)

                    # get indices to split on. remove last because it is empty
                    split_indices = np.cumsum([len(i) for i in radius_ranges])[:-1]
                    # slice arrays by radii
                    pos_arrays = np.split(pos, split_indices, axis= -1 )

                    if not continuous:
                      agree_arrays = np.split(agree, split_indices, axis = -1)
                      similarities = np.stack(list(map(lambda x: np.average(x, axis = -1), pos_arrays + agree_arrays)),axis=1)
                    else:
                      # don't use agreement features when there are continuous values
                      similarities = np.stack(list(map(lambda x: np.average(x, axis = -1), pos_arrays)),axis=1)
                else:
                    # no radius, so no similarities. just an empty placeholder
                    similarities = np.zeros((len(eval_cell_types),0,0))

                # reshape similarities to flatten 1st dimension, which are the targets
                # results in the odering:
                ## row 1: cell 1: pos for each target and agree for each target for each radius
                similarities = similarities.reshape(similarities.shape[0], similarities.shape[1]*similarities.shape[2])

                ##### Concatenate all cell type features together ####
                final_features = np.concatenate([data[feature_cell_indices,i], similarities],axis=1).flatten()

                # mask missing data
                f_mask = np.concatenate([feature_cell_indices!=-1,
                                         np.ones(similarities.shape)],axis=1).flatten()
                final_features = final_features[f_mask != 0]

                if (mode != Dataset.RUNTIME):
                    labels = data[label_cell_indices_no_similarities,i]

                else: # used when just predicting
                    # The features going into the example.
                    labels = garbage_labels # all 0's

                # append labels and targetmask
                final= tuple([final_features, labels.astype(np.float32), target_mask.astype(np.float32)])

                #### Finish appending feature labels together ####
                if (return_feature_names):
                    all_labels = []
                    feature_names = []
                    similarity_labels_dp = ['r%i_%s' % (radius, 'dp') for radius in radii]
                    if continuous:
                      similarity_labels = similarity_labels_dp
                    else:
                      similarity_labels_agreement = ['r%i_%s' % (radius, 'agree') for radius in radii]
                      similarity_labels = np.concatenate([similarity_labels_dp, similarity_labels_agreement])

                    # concatenate together feature names
                    for j,c in enumerate(eval_cell_types):
                        tmp = np.array(feature_targets)[feature_cell_indices[j,:] != -1]
                        al = ['%s_%s' % (c, a) for a in tmp]
                        sl = ['%s_%s' % (c, s) for s in similarity_labels]

                        feature_names.append(al)
                        feature_names.append(sl)

                    all_labels.append(np.concatenate(feature_names))
                    all_labels.append(['lbl_%s_%s' % (cell, a) for a in label_targets]) # of form lbl_cellline_target
                    all_labels.append(['mask_%s_%s' % (cell, a) for a in label_targets]) # of form mask_cellline_target

                    yield (final, tuple(all_labels))
                else:
                    yield final


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
    target_mask = f[-1]
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
