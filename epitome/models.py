r"""
======
Models
======
.. currentmodule:: epitome.models

.. autosummary::
  :toctree: _generate/

  PeakModel
  EpitomeModel
"""


from epitome import *
import tensorflow as tf
import pandas as pd

from .functions import *
from .constants import Dataset
from .generators import generator_to_tf_dataset,load_data
from .dataset import EpitomeDataset
from .metrics import *
from .conversion import *
import numpy as np

import tqdm
import logging
import sys
import os

# for saving model
import pickle
from operator import itemgetter

#######################################################################
#################### Variational Peak Model ###########################
#######################################################################

class PeakModel():
    """
    Model for learning from ChIP-seq peaks.
    """

    def __init__(self,
                 dataset,
                 test_celltypes = [],
                 single_cell = False,
                 debug = False,
                 batch_size = 64,
                 shuffle_size = 10,
                 prefetch_size = 10,
                 l1=0.,
                 l2=0.,
                 lr=1e-3,
                 radii=[1,3,10,30],
                 checkpoint = None,
                 max_valid_batches = None):
        '''
        Initializes Peak Model

        :param EpitomeDataset dataset: EpitomeDataset
        :param list test_celltypes: list of cell types to hold out for test. Should be in cellmap
        :param boolean single_cell: whether you are building a model to predict using scATAC-seq posteriors. Defaults to False.
        :param bool debug: used to print out intermediate validation values
        :param int batch_size: batch size (default is 64)
        :param int shuffle_size: data shuffle size (default is 10)
        :param int prefetch_size: data prefetch size (default is 10)
        :param floatl1: l1 regularization (default is 0)
        :param float l2: l2 regularization (default is 0)
        :param float lr: lr (default is 1e-3)
        :param list radii: radius of DNase-seq to consider around a peak of interest (default is [1,3,10,30]) each model.
        :param str checkpoint: path to load model from.
        :param int max_valid_batches: the size of train-validation dataset (default is None, meaning that it doesn't create a train-validation dataset or stop early while training)
        '''

        logging.getLogger("tensorflow").setLevel(logging.INFO)

        # set the dataset
        self.dataset = dataset
        # whether or not this model can train using continuous datas
        self.single_cell = single_cell

        assert (set(test_celltypes) < set(list(self.dataset.cellmap))), \
                "test_celltypes %s must be subsets of available cell types %s" % (str(test_celltypes), str(list(dataset.cellmap)))

        # get evaluation cell types by removing any cell types that would be used in test
        self.eval_cell_types = list(self.dataset.cellmap).copy()
        self.test_celltypes = test_celltypes
        [self.eval_cell_types.remove(test_cell) for test_cell in self.test_celltypes]

        if max_valid_batches is not None:
            # get the last training chromosome if valid_chromosome is not specified
            tmp_chrs = self.dataset.regions.chromosomes
            for i in self.dataset.test_chrs:
                tmp_chrs.remove(i)
            for i in self.dataset.valid_chrs:
                tmp_chrs.remove(i)

            valid_chromosome = tmp_chrs[-1]

            # Reserve chromosome 22 from the training data to validate model while training
            self.dataset.set_train_validation_indices(valid_chromosome)
            print(self.dataset.get_data(Dataset.TRAIN_VALID).shape)

            # Creating a separate train-validation dataset
            _, _, self.train_valid_iter = generator_to_tf_dataset(load_data(self.dataset.get_data(Dataset.TRAIN_VALID),
                                                    self.eval_cell_types,
                                                    self.eval_cell_types,
                                                    dataset.matrix,
                                                    dataset.targetmap,
                                                    dataset.cellmap,
                                                    continuous = self.single_cell,
                                                    similarity_targets= dataset.similarity_targets,
                                                    radii = radii, mode = Dataset.TRAIN),
                                                    batch_size, shuffle_size, prefetch_size)

        input_shapes, output_shape, self.train_iter = generator_to_tf_dataset(load_data(self.dataset.get_data(Dataset.TRAIN),
                                                self.eval_cell_types,
                                                self.eval_cell_types,
                                                dataset.matrix,
                                                dataset.targetmap,
                                                dataset.cellmap,
                                                continuous = self.single_cell,
                                                similarity_targets = dataset.similarity_targets,
                                                radii = radii, mode = Dataset.TRAIN),
                                                batch_size, shuffle_size, prefetch_size)

        _, _,            self.valid_iter = generator_to_tf_dataset(load_data(self.dataset.get_data(Dataset.VALID),
                                                self.eval_cell_types,
                                                self.eval_cell_types,
                                                dataset.matrix,
                                                dataset.targetmap,
                                                dataset.cellmap,
                                                continuous = self.single_cell,
                                                similarity_targets = dataset.similarity_targets,
                                                radii = radii, mode = Dataset.VALID),
                                                batch_size, 1, prefetch_size)

        # can be empty if len(test_celltypes) == 0
        if len(self.test_celltypes) > 0:
            _, _,            self.test_iter = generator_to_tf_dataset(load_data(self.dataset.get_data(Dataset.TEST),
                                                   self.test_celltypes,
                                                   self.eval_cell_types,
                                                   dataset.matrix,
                                                   dataset.targetmap,
                                                   dataset.cellmap,
                                                   continuous = self.single_cell,
                                                   similarity_targets = dataset.similarity_targets,
                                                   radii = radii, mode = Dataset.TEST),
                                                   batch_size, 1, prefetch_size)

        self.l1, self.l2 = l1, l2
        self.lr = lr
        self.batch_size = batch_size
        self.prefetch_size = prefetch_size
        self.shuffle_size = shuffle_size
        self.optimizer =tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr)

        self.num_outputs = output_shape[0]
        self.num_inputs = input_shapes
        self.max_valid_batches = max_valid_batches

        # set self
        self.radii = radii
        self.debug = debug
        self.model = self.create_model()

    def save(self, checkpoint_path):
        '''
        Saves model.

        :param str checkpoint_path: string file path to save model to.
        '''

        weights_path = os.path.join(checkpoint_path, "weights.h5")
        meta_path = os.path.join(checkpoint_path, "model_params.pickle")

        # save keras model weights
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        file = h5py.File(weights_path, 'w')
        weight = self.model.get_weights()
        for i in range(len(weight)):
            file.create_dataset('weight' + str(i), data=weight[i])
        file.close()

        # save model params to pickle file
        dict_ = {'dataset_params': self.dataset.get_parameter_dict(),
                         'test_celltypes':self.test_celltypes,
                         'debug': self.debug,
                         'batch_size':self.batch_size,
                         'single_cell': self.single_cell,
                         'shuffle_size':self.shuffle_size,
                         'prefetch_size':self.prefetch_size,
                         'radii':self.radii}

        fileObject = open(meta_path,'wb')
        pickle.dump(dict_,fileObject)
        fileObject.close()

    def body_fn(self):
        raise NotImplementedError()

    def g(self, p, a=1, B=0, y=1):
        '''
        Normalization Function. Normalizes loss w.r.t. label proportion.

        Constraints:
         1. g(p) = 1 when p = 1
         2. g(p) = a * p^y + B, where a, y and B are hyperparameters

         :param int p: base
         :param int a: constant multiplier
         :param int B: additive constant
         :param int y: power
         :return: normalized loss
         :rtype: float

        '''
        return a * tf.math.pow(p, y) + B

    def loss_fn(self, y_true, y_pred, weights):
        '''
        Loss function for Epitome. Calculates the weighted sigmoid cross entropy
        between logits and true values.

        :param tensor or numpy.array y_true: true binary values
        :param tensor or numpy.array y_pred: logits
        :param tensor or numpy.array weights: binary weights whether the true values exist for
            a given cell type/target combination
        :return: Loss summed over all TFs and genomic loci.
        :rtype: tensor
        '''
        # weighted sum of cross entropy for non 0 weights
        # Reduction method = Reduction.SUM_BY_NONZERO_WEIGHTS
        loss = tf.compat.v1.losses.sigmoid_cross_entropy(y_true,
                                                        y_pred,
                                                        weights = weights,
                                                        reduction = tf.compat.v1.losses.Reduction.NONE)

        return tf.math.reduce_sum(loss, axis=0)

    def train(self, max_train_batches, patience=3, min_delta=0.01):
        '''
        Trains an Epitome model. If the patience and min_delta are not specified, the model will train on max_train_batches.
        Else, the model will either train on max_train_batches or stop training early if the train_valid_loss is converging
        (based on the patience and/or min_delta hyper-parameters), whatever comes first.

        :param int max_train_batches: max number of batches to train for
        :param int patience: number of train-valid iterations (200 batches) with no improvement after which training will be stopped.
        :param float min_delta: minimum change in the monitored quantity to qualify as an improvement,
          i.e. an absolute change of less than min_delta, will count as no improvement.

        :return triple of number of batches trained for the best model, number of batches the model has trained total,
          the train_validation losses (returns an empty list if self.max_valid_batches is None).
        :rtype: tuple
        '''
        tf.compat.v1.logging.info("Starting Training")

        @tf.function
        def train_step(f):
            features = f[:-2]
            labels = f[-2]
            weights = f[-1]

            with tf.GradientTape() as tape:

                logits = self.model(features, training=True)
                loss = self.loss_fn(labels, logits, weights)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            return loss

        @tf.function
        def valid_step(f):
            features = f[:-2]
            labels = f[-2]
            weights = f[-1]
            logits = self.model(features, training=False)
            neg_log_likelihood = self.loss_fn(labels, logits, weights)
            return neg_log_likelihood

        def loopiter():
            # Initializing variables
            mean_valid_loss, decreasing_train_valid_iters, best_model_batches = sys.maxsize, 0, 0
            train_valid_losses = []

            for current_batch, f in enumerate(self.train_iter):
                loss = train_step(f)

                if current_batch % 1000 == 0:
                  tf.print("Batch: ", current_batch)
                  tf.print("\tLoss: ", tf.reduce_mean(loss))

                if (current_batch % 200 == 0) and (self.max_valid_batches is not None):
                    # Early Stopping Validation
                    new_valid_loss = []

                    for current_valid_batch, f_v in enumerate(self.train_valid_iter):
                        new_valid_loss.append(valid_step(f_v))

                        if (current_valid_batch == self.max_valid_batches):
                            break

                    new_mean_valid_loss = tf.reduce_mean(new_valid_loss)
                    train_valid_losses.append(new_mean_valid_loss)


                    # Check if the improvement in train-validation loss is at least min_delta.
                    # If the train-validation loss has increased more than patience consecutive train-validation iterations, the function stops early.
                    # Else it continues training.
                    improvement = mean_valid_loss - new_mean_valid_loss
                    if improvement < min_delta:
                        decreasing_train_valid_iters += 1
                        if decreasing_train_valid_iters == patience:
                            return best_model_batches, current_batch, train_valid_losses
                    else:
                        # If the train-validation loss decreases (the model is improving and converging), reset decreasing_train_valid_iters and mean_valid_loss.
                        decreasing_train_valid_iters = 0
                        mean_valid_loss = new_mean_valid_loss
                        best_model_batches = current_batch

                if (current_batch >= max_train_batches):
                    break

            return best_model_batches, max_train_batches, train_valid_losses

        return loopiter()

    def test(self, num_samples, mode = Dataset.VALID, calculate_metrics=False):
        '''
        Tests model on valid and test dataset handlers.

        :param int num_samples: number of data points to run on
        :param Dataset Enum mode: what mode to run in DATASET.VALID, TRAIN, or TEST)
        :param bool calculate_metrics: whether to return auROC/auPR
        :return: dictionary of results
        :rtype: dict
        '''

        if (mode == Dataset.VALID):
            handle = self.valid_iter # for standard validation of validation cell types
        elif (mode == Dataset.TRAIN):
            handle = self.train_valid_iter
        elif (mode == Dataset.TEST and len(self.test_celltypes) > 0):
            handle = self.test_iter # for standard validation of validation cell types
        else:
            raise Exception("No data exists for %s. Use function test_from_generator() if you want to create a new iterator." % (mode))

        return self.run_predictions(num_samples, handle, calculate_metrics)

    def test_from_generator(self, num_samples, ds, calculate_metrics=True):
        '''
        Runs test given a specified data generator.

        :param int num_samples: number of samples to test
        :param tensorflow dataset ds: tensorflow dataset, created by dataset_to_tf_dataset
        :param bool calculate_metrics: whether to return auROC/auPR
        :return: predictions
        :rtype: dict
        '''

        return self.run_predictions(num_samples, ds, calculate_metrics)

    def eval_vector(self, matrix, indices):
        '''
        Evaluates a new cell type based on its chromatin (DNase or ATAC-seq) vector, as well
        as any other similarity targets (acetylation, methylation, etc.). len(vector) should equal
        the self.dataset.get_data(Dataset.ALL).shape[1]

        :param numpy.matrix matrix: matrix of 0s/1s, where # rows match # similarity targets in model
        :param numpy.array indices: indices of vector to actually score. You need all of the locations for the generator.
        :param int samples: number of times to sample from network
        :return: predictions
        :rtype: dict
        '''

        # verify matrix has correct number of genomic positions
        assert matrix.shape[-1] == len(self.dataset.regions), \
          """Error: number of columns in matrix must match genomic regions in dataset.regions.
          But got matrix.shape[-1]=%i and len(self.dataset.regions)=%i
          """  % (matrix.shape[-1], len(self.dataset.regions))

        # find regions that have some signal (sum > 0)
        region_sums = np.sum(self.dataset.get_data(Dataset.ALL)[:,indices], axis=0)

        # only pick indices that have some signal
        nonzero_indices = np.where(region_sums > 0)[0]
        filtered_indices = indices[nonzero_indices]

        input_shapes, output_shape, ds = generator_to_tf_dataset(load_data(self.dataset.get_data(Dataset.ALL),
                 self.test_celltypes,   # used for labels. Should be all for train/eval and subset for test
                 self.eval_cell_types,   # used for rotating features. Should be all - test for train/eval
                 self.dataset.matrix,
                 self.dataset.targetmap,
                 self.dataset.cellmap,
                 continuous = self.single_cell,
                 radii = self.radii,
                 mode = Dataset.RUNTIME,
                 similarity_matrix = matrix,
                 similarity_targets = self.dataset.similarity_targets,
                 indices = filtered_indices), self.batch_size, 1, self.prefetch_size)

        num_samples = len(filtered_indices)
        results = self.run_predictions(num_samples, ds, calculate_metrics = False)['preds']

        # mix back in filtered_indices with original indices
        all_results = np.empty((indices.shape[0], results.shape[-1]))

        all_results[:] = np.nan # set missing values to nan
        all_results[nonzero_indices,:] = results

        return all_results

    @tf.function
    def predict_step_matrix(self, inputs):
        """
        Runs predictions on numpy matrix from _predict

        :param np.ndarray inputs: numpy matrix of examples x features
        :return: predictions
        :rtype: np.ndarray
        """

        # predict
        return self.predict_step_generator(inputs)

    def _predict(self, numpy_matrix):
        '''
        Run predictions on a numpy matrix. Size of numpy_matrix should be # examples by features.
        This function is mostly used for testing, as it requires the user to pre-generate the
        features using the generator function in generators.py.

        :param numpy.matrix numpy_matrix: matrix of features to predict
        :return: predictions
        :rtype: tensor
        '''
        return self.predict_step_matrix(numpy_matrix)

    @tf.function
    def predict_step_generator(self, inputs_b):
        '''
        Runs predictions on inputs from run_predictions

        :param tf.Tensor inputs_b: batch of input data
        :return: predictions
        :rtype: tuple
        '''
        return tf.sigmoid(self.model(inputs_b))

    def run_predictions(self, num_samples, iter_, calculate_metrics = True):
        '''
        Runs predictions on num_samples records

        :param int num_samples: number of samples to test
        :param tf.dataset iter_: output of self.sess.run(generator_to_one_shot_iterator()), handle to one shot iterator of records
        :param bool calculate_metrics: whether to return auROC/auPR

        :return: dict of preds, truth, target_dict, auROC, auPRC, False
            preds = predictions,
            truth = actual values,
            sample_weight: 0/1 weights on predictions.
            target_dict = if log=True, holds predictions for individual factors
            auROC = average macro area under ROC for all factors with truth values
            auPRC = average area under PRC for all factors with truth values
        :rtype: dict
        '''

        batches = int(num_samples / self.batch_size)+1

        # empty arrays for concatenation
        truth = []
        preds = []
        sample_weight = []

        for f in tqdm.tqdm(iter_.take(batches)):
            inputs_b = f[:-2]
            truth_b = f[-2]
            weights_b = f[-1]
            preds_b = self.predict_step_generator(inputs_b)

            preds.append(preds_b)
            truth.append(truth_b)
            sample_weight.append(weights_b)

        # concat all results
        preds = tf.concat(preds, axis=0)

        truth = tf.concat(truth, axis=0)
        sample_weight = tf.concat(sample_weight, axis=0)

        # trim off extra from last batch
        truth = truth[:num_samples, :].numpy()
        preds = preds[:num_samples, :].numpy()
        sample_weight = sample_weight[:num_samples, :].numpy()

        # reset truth back to 0 to compute metrics
        # sample weights will rule these out anyways when computing metrics
        truth_reset = np.copy(truth)
        truth_reset[truth_reset < Label.UNBOUND.value] = 0

        # do not continue to calculate metrics. Just return predictions and true values
        if (not calculate_metrics):
            return {
                'preds': preds,
                'truth': truth,
                'weights': sample_weight,
                'target_dict': None,
                'auROC': None,
                'auPRC': None
            }

        assert(preds.shape == sample_weight.shape)

        try:

            # try/accept for cases with only one class (throws ValueError)
            target_dict = get_performance(self.dataset.targetmap, preds, truth_reset, sample_weight, self.dataset.predict_targets)

            # calculate averages
            auROC = np.nanmean(list(map(lambda x: x['AUC'],target_dict.values())))
            auPRC = np.nanmean(list(map(lambda x: x['auPRC'],target_dict.values())))

            tf.compat.v1.logging.info("macro auROC:     " + str(auROC))
            tf.compat.v1.logging.info("auPRC:     " + str(auPRC))
        except ValueError:
            auROC = None
            auPRC = None
            tf.compat.v1.logging.info("Failed to calculate metrics")

        return {
            'preds': preds,
            'truth': truth,
            'weights': sample_weight,
            'target_dict': target_dict,
            'auROC': auROC,
            'auPRC': auPRC
        }

    def score_whole_genome(self, similarity_peak_files,
                       file_prefix,
                       chrs=None):
        '''
        Runs a whole genome scan for all available genomic regions in the dataset (about 3.2Million regions)
        Takes about 1 hour on entire genome.

        :param list similarity_peak_files: list of similarity_peak_files corresponding to similarity_targets
        :param str file_prefix: path to save compressed numpy file to. Adds '.npz' extension.
        :param list chrs: list of chromosome names to score. If none, scores all chromosomes.
        '''

        restructured_similarity = []
        for f in similarity_peak_files:
            tmpConversion = RegionConversion(self.dataset.regions, f)
            restructured_similarity.append(tmpConversion.get_binary_vector()[0])

        peak_matrix = np.vstack(restructured_similarity)

        # get sorted indices to score
        regions  = self.dataset.regions

        # filter regions by chrs
        if chrs is not None:
            regions = regions[regions.Chromosome.isin(chrs)]

        idx = np.array(sorted(list(regions.idx)))

        print("scoring %i regions" % idx.shape[0])

        predictions = self.eval_vector(peak_matrix, idx)
        print("finished predictions...", predictions.shape)

        # return matrix of region, TF information. trim off idx column
        npRegions =  regions.df.sort_values(by='idx').drop(labels='idx', axis=1).values

        # TODO turn into right types (all strings right now)
        # predictions[0] is means of size n regions by # ChIP-seq peaks predicted
        preds = np.concatenate([npRegions, predictions], axis=1)

        # can load back in using:
        # > loaded = np.load('file_prefix.npz')
        # > loaded['preds']
        # TODO: save the right types!  (currently all strings!)
        np.savez_compressed(file_prefix, preds = preds,
                            names=np.array(['chr','start','end'] + self.dataset.predict_targets))

        print("columns for matrices are chr, start, end, %s" % ", ".join(self.dataset.predict_targets))

    def score_matrix(self, accessilibility_peak_matrix, regions):
        """ Runs predictions on a matrix of accessibility peaks, where columns are samples and
        rows are regions from regions_peak_file. rows in accessilibility_peak_matrix should matching

        :param numpy.matrix accessilibility_peak_matrix:  of (samples by genomic regions)
        :param str regions: either narrowpeak or bed file containing regions to score, OR a pyranges object
            with columns [Chomosome, Start, End, idx]. Index matches each genomic region to a row in
            accessilibility_peak_matrix. In both cases, number of regions Should
            match rows in accessilibility_peak_matrix

        :return: 3-dimensional numpy matrix of predictions: sized (samples by regions by ChIP-seq targets)
        :rtype: numpy matrix
        """

        conversionObject = RegionConversion(self.dataset.regions, regions)

        results = []

        # TODO 9/10/2020: should do something more efficiently than a for loop
        for sample_i in tqdm.tqdm(range(accessilibility_peak_matrix.shape[0])):

            peaks_i, idx = conversionObject.get_binary_vector(vector = accessilibility_peak_matrix[sample_i,:])

            preds = self.eval_vector(peaks_i, idx)

            # group preds by joined['idx']
            results.append(preds)

        # stack all samples along 0th axis
        # shape: samples x regions x TFs
        tmp = np.stack(results)

        # mean and merge along 1st axis
        return conversionObject.merge(tmp, axis = 1)


    def score_peak_file(self, similarity_peak_files, regions_peak_file):
        '''
        Runs predictions on a set of peaks defined in a bed or narrowPeak file.

        :param list similarity_peak_files: list of narrowpeak or bed files containing peaks for similarity assays.
          Length(similarity_peak_files) should equal the number of similarity_targets in self.dataset.
        :param str regions_peak_file: narrowpeak or bed file containing regions to score.
        :return: pandas dataframe of genomic regions and predictions
        :rtype: pandas.dataframe
        '''

        # get peak_vector, which is a vector matching train set. Some peaks will not overlap train set,
        # and their indices are stored in missing_idx for future use
        restructured_similarity = []
        for f in similarity_peak_files:
          tmpConversion = RegionConversion(self.dataset.regions, f)
          restructured_similarity.append(tmpConversion.get_binary_vector()[0])
        peak_matrix = np.vstack(restructured_similarity)

        # only select peaks to score
        compareObject = RegionConversion(self.dataset.regions, regions_peak_file)
        # get indices in self.dataset.regions that we will be scoring
        idx = compareObject.get_base_overlap_index()

        print("scoring %i regions" % idx.shape[0])

        if len(idx) == 0:
            raise ValueError("No positive peaks found in %s" % regions_peak_file)

        preds = self.eval_vector(peak_matrix, idx)
        print("finished predictions...", preds.shape)

        if not isinstance(preds, np.ndarray):
            preds = preds.numpy()

        merged_preds = compareObject.merge(preds, axis = 0) # TODO is this the correct axis?

        preds_df = pd.DataFrame(data=merged_preds, columns=self.dataset.predict_targets)
        return pd.concat([compareObject.compare_df(), preds_df], axis=1)


class EpitomeModel(PeakModel):
    def __init__(self,
             *args,
             **kwargs):
        '''
        Creates a new model with 4 layers with 100 unites each.
            To resume model training on an old model, call:

            .. code-block:: python

                model = EpitomeModel(checkpoint=path_to_saved_model)
        '''
        self.activation = tf.tanh
        self.layers = 2

        if "checkpoint" in kwargs.keys():
            fileObject = open(kwargs["checkpoint"] + "/model_params.pickle" ,'rb')
            metadata = pickle.load(fileObject)
            fileObject.close()

            # reconstruct dataset and delete unused parameters
            dataset = EpitomeDataset(**metadata['dataset_params'])
            metadata['dataset'] = dataset
            del metadata['dataset_params']

            PeakModel.__init__(self, **metadata, **kwargs)
            file = h5py.File(os.path.join(kwargs["checkpoint"], "weights.h5"), 'r')

            # load model weights back in
            weights = []
            for i in range(len(file.keys())):
                weights.append(file['weight' + str(i)][:])
            self.model.set_weights(weights)
            file.close()

        else:
            PeakModel.__init__(self, *args, **kwargs)

    def create_model(self, **kwargs):
        '''
        Creates an Epitome model.
        '''
        cell_inputs = [tf.keras.layers.Input(shape=(self.num_inputs[i],))
                       for i in range(len(self.num_inputs))]

        # make a channel for each cell type
        cell_channels = []

        for i in range(len(self.num_inputs)):
            # make input layer for cell
            last = cell_inputs[i]
            for j in range(self.layers):
                num_units = int(self.num_inputs[i]/(2 * (j+1)))
                d = tf.keras.layers.Dense(num_units,
                                                activation = self.activation)(last)
                last = d
            cell_channels.append(last)


        # merge together all cell channels
        if (len(cell_channels) > 1):
            last = tf.keras.layers.concatenate(cell_channels)
        else:
            last = cell_channels[0]

        outputs =  tf.keras.layers.Dense(self.num_outputs,
                                        activity_regularizer=tf.keras.regularizers.l1_l2(self.l1, self.l2),
                                        name="output_layer")(last)

        model = tf.keras.models.Model(inputs=cell_inputs, outputs=outputs)

        return model
