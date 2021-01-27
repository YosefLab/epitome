r"""
======
Models
======
.. currentmodule:: epitome.models

.. autosummary::
  :toctree: _generate/

  VariationalPeakModel
  VLP
"""


from epitome import *
import tensorflow as tf
import tensorflow_probability as tfp

from .functions import *
from .constants import *
from .generators import *
from .dataset import *
from .metrics import *
import numpy as np

import tqdm
import logging

# for saving model
import pickle
from operator import itemgetter

#######################################################################
#################### Variational Peak Model ###########################
#######################################################################

class VariationalPeakModel():
    """
    Model for learning from ChIP-seq peaks.
    Modeled from `this Bayesian Neural Network <https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/bayesian_neural_network.py>`_.
    """

    def __init__(self,
                 dataset,
                 test_celltypes = [],
                 debug = False,
                 batch_size = 64,
                 shuffle_size = 10,
                 prefetch_size = 10,
                 l1=0.,
                 l2=0.,
                 lr=1e-3,
                 radii=[1,3,10,30],
                 checkpoint = None):
        '''
        Initializes Peak Model

        :param EpitomeDataset dataset: EpitomeDataset
        :param list test_celltypes: list of cell types to hold out for test. Should be in cellmap
        :param bool debug: used to print out intermediate validation values
        :param int batch_size: batch size (default is 64)
        :param int shuffle_size: data shuffle size (default is 10)
        :param int prefetch_size: data prefetch size (default is 10)
        :param floatl1: l1 regularization (default is 0)
        :param float l2: l2 regularization (default is 0)
        :param float lr: lr (default is 1e-3)
        :param list radii: radius of DNase-seq to consider around a peak of interest (default is [1,3,10,30])
        each model.
        :param str checkpoint: path to load model from.
        '''

        logging.getLogger("tensorflow").setLevel(logging.INFO)

        # set the dataset
        self.dataset = dataset

        assert (set(test_celltypes) < set(list(self.dataset.cellmap))), \
                "test_celltypes %s must be subsets of available cell types %s" % (str(test_celltypes), str(list(dataset.cellmap)))

        # get evaluation cell types by removing any cell types that would be used in test
        self.eval_cell_types = list(self.dataset.cellmap).copy()
        self.test_celltypes = test_celltypes
        [self.eval_cell_types.remove(test_cell) for test_cell in self.test_celltypes]

        input_shapes, output_shape, self.train_iter = generator_to_tf_dataset(load_data(self.dataset.get_data(Dataset.TRAIN),
                                                self.eval_cell_types,
                                                self.eval_cell_types,
                                                dataset.matrix,
                                                dataset.targetmap,
                                                dataset.cellmap,
                                                similarity_targets = dataset.similarity_targets,
                                                radii = radii, mode = Dataset.TRAIN),
                                                batch_size, shuffle_size, prefetch_size)

        _, _,            self.valid_iter = generator_to_tf_dataset(load_data(self.dataset.get_data(Dataset.VALID),
                                                self.eval_cell_types,
                                                self.eval_cell_types,
                                                dataset.matrix,
                                                dataset.targetmap,
                                                dataset.cellmap,
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

        # set self
        self.radii = radii
        self.debug = debug
        self.model = self.create_model()

    def get_weight_parameters(self):
        '''
        Extracts weight posterior statistics for layers with weight distributions.

        :param keras model model: keras model
        :return: triple of layer names, weight means for each layer and stddev for each layer.
        :rtype: tuple
        '''

        names = []
        qmeans = []
        qstds = []
        for i, layer in enumerate(self.model.layers):
            try:
                q = layer.kernel_posterior
            except AttributeError:
                continue
            names.append("Layer {}".format(i))
            qmeans.append(q.mean())
            qstds.append(q.stddev())

        return (names, qmeans, qstds)

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

    def train(self, num_steps):
        '''
        Trains an Epitome model for num_steps iterations.

        :param int num_steps: number of iterations to train for
        '''

        tf.compat.v1.logging.info("Starting Training")

        @tf.function
        def train_step(f):
            features = f[:-2]
            labels = f[-2]
            weights = f[-1]

            with tf.GradientTape() as tape:

                logits = self.model(features, training=True)
                kl_loss = tf.reduce_sum(self.model.losses)
                neg_log_likelihood = self.loss_fn(labels, logits, weights)
                elbo_loss = neg_log_likelihood + kl_loss

            gradients = tape.gradient(elbo_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            return elbo_loss, neg_log_likelihood, kl_loss

        @tf.function
        def loopiter():
            for step, f in enumerate(self.train_iter):
                loss = train_step(f)

                if step % 100 == 0:
                  tf.compat.v1.logging.info(str(step) + " " + str(tf.reduce_mean(loss[0])) +
                                            str(tf.reduce_mean(loss[1])) +
                                            str(tf.reduce_mean(loss[2])))

                  if (self.debug):
                      tf.compat.v1.logging.info("On validation")
                      _, _, _, _, _ = self.test(40000, log=False)
                      tf.compat.v1.logging.info("")

                if step > num_steps:
                  break

        loopiter()

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

    def eval_vector(self, matrix, indices, samples = 50):
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

        input_shapes, output_shape, ds = generator_to_tf_dataset(load_data(self.dataset.get_data(Dataset.ALL),
                 self.test_celltypes,   # used for labels. Should be all for train/eval and subset for test
                 self.eval_cell_types,   # used for rotating features. Should be all - test for train/eval
                 self.dataset.matrix,
                 self.dataset.targetmap,
                 self.dataset.cellmap,
                 radii = self.radii,
                 mode = Dataset.RUNTIME,
                 similarity_matrix = matrix,
                 similarity_targets = self.dataset.similarity_targets,
                 indices = indices), self.batch_size, 1, self.prefetch_size)

        num_samples = len(indices)

        results = self.run_predictions(num_samples, ds, calculate_metrics = False)

        return results['preds_mean'], results['preds_std']

    @tf.function
    def predict_step_matrix(self, inputs):
        """
        Runs predictions on numpy matrix from _predict

        :param np.ndarray inputs: numpy matrix of examples x features
        :return: mean of predictions
        :rtype: np.ndarray
        """

        # get the shapes for the cell type specific features.
        # they are not even, because some cells have missing data.
        cell_lens = [i.shape[-1] for i in self.train_iter.element_spec[:-2]]

        # split matrix inputs into tuple of cell line specific features
        split_inputs = tf.split(tf.dtypes.cast(inputs, tf.float32), cell_lens, axis=1)

        # predict
        return predict_step_generator(split_inputs, samples=1)[0]

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
    def predict_step_generator(self, inputs_b, samples = 50):
        """Runs predictions on inputs from run_predictions

        :param tf.Tensor inputs_b: batch of input data
        :param int samples: Number of samples to test. Defaults to 50.
        :return: mean and standard deviations of predictions
        :rtype: tuple
        """

        # sample n times by tiling batch by rows, running
        # predictions for each row
        inputs_tiled = [tf.tile(i, (samples, 1)) for i in inputs_b]
        tmp = self.model(inputs_tiled)
        y_pred = tf.sigmoid(tmp)
        # split up batches into a third dimension and stack them in third dimension
        preds = tf.stack(tf.split(y_pred, samples, axis=0), axis=0)
        return tf.math.reduce_mean(preds, axis=0), tf.math.reduce_std(preds, axis=0)

    def run_predictions(self, num_samples, iter_, calculate_metrics = True, samples = 50):
        '''
        Runs predictions on num_samples records

        :param int num_samples: number of samples to test
        :param tf.dataset iter_: output of self.sess.run(generator_to_one_shot_iterator()), handle to one shot iterator of records
        :param bool calculate_metrics: whether to return auROC/auPR
        :param int samples: number of times to sample from network

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
        preds_mean = []
        preds_std = []
        sample_weight = []

        for f in tqdm.tqdm(iter_.take(batches)):
            inputs_b = f[:-2]
            truth_b = f[-2]
            weights_b = f[-1]
            # Calculate epistemic uncertainty for batch by iterating over a certain number of times,
            # getting y_preds. You can then calculate the mean and sigma of the predictions,
            # and use this to gather uncertainty: (see http://krasserm.github.io/2019/03/14/bayesian-neural-networks/)
            # inputs, truth, sample_weight
            preds_mean_b, preds_std_b = self.predict_step_generator(inputs_b, samples)

            preds_mean.append(preds_mean_b)
            preds_std.append(preds_std_b)
            truth.append(truth_b)
            sample_weight.append(weights_b)

        # concat all results
        preds_mean = tf.concat(preds_mean, axis=0)
        preds_std = tf.concat(preds_std, axis=0)

        truth = tf.concat(truth, axis=0)
        sample_weight = tf.concat(sample_weight, axis=0)

        # trim off extra from last batch
        truth = truth[:num_samples, :].numpy()
        preds_mean = preds_mean[:num_samples, :].numpy()
        preds_std = preds_std[:num_samples, :].numpy()
        sample_weight = sample_weight[:num_samples, :].numpy()

        # reset truth back to 0 to compute metrics
        # sample weights will rule these out anyways when computing metrics
        truth_reset = np.copy(truth)
        truth_reset[truth_reset < Label.UNBOUND.value] = 0

        # do not continue to calculate metrics. Just return predictions and true values
        if (not calculate_metrics):
            return {
                'preds_mean': preds_mean,
                'preds_std': preds_std,
                'truth': truth,
                'weights': sample_weight,
                'target_dict': None,
                'auROC': None,
                'auPRC': None
            }

        assert(preds_mean.shape == sample_weight.shape)

        try:

            # try/accept for cases with only one class (throws ValueError)
            target_dict = get_performance(self.dataset.targetmap, preds_mean, truth_reset, sample_weight, self.dataset.predict_targets)

            # calculate averages
            auROC = np.nanmean(list(map(lambda x: x['AUC'],target_dict.values())))
            auPRC = np.nanmean(list(map(lambda x: x['auPRC'],target_dict.values())))

            tf.compat.v1.logging.info("macro auROC:     " + str(auROC))
            tf.compat.v1.logging.info("auPRC:     " + str(auPRC))
        except ValueError as v:
            auROC = None
            auPRC = None
            tf.compat.v1.logging.info("Failed to calculate metrics")

        return {
            'preds_mean': preds_mean,
            'preds_std': preds_std,
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

        # get peak_vector, which is a vector matching train set. Some peaks will not overlap train set,
        # and their indices are stored in missing_idx for future use
        peak_vectors = [pyranges2Vector(bed2Pyranges(f), self.dataset.regions)[0] for f in similarity_peak_files]
        peak_matrix = np.vstack(peak_vectors)
        del peak_vectors

        regions  = self.dataset.regions

        # filter regions by chrs
        if chrs is not None:
            regions = regions[regions.Chromosome.isin(chrs)]


        # get sorted indices to score
        idx = np.array(sorted(list(regions.idx)))

        print("scoring %i regions" % idx.shape[0])

        # tuple of means and stds
        predictions = self.eval_vector(peak_matrix, idx)
        print("finished predictions...", predictions[0].shape)

        # zip together means for each position in idx

        # return matrix of region, TF information. trim off idx column
        npRegions = regions.df.sort_values(by='idx').values[:,:-1]
        # TODO turn into right types (all strings right now)
        # predictions[0] is means of size n regions by # ChIP-seq peaks predicted
        means = np.concatenate([npRegions, predictions[0]], axis=1)

        # can load back in using:
        # > loaded = np.load('file_prefix.npz')
        # > loaded['means'], loaded['stds']
        # TODO: save the right types!  (currently all strings!)
        np.savez_compressed(file_prefix, means = means,
                            names=np.array(['chr','start','end'] + self.dataset.predict_targets))

        print("columns for matrices are chr, start, end, %s" % ", ".join(self.dataset.predict_targets))

    def score_matrix(self, accessilibility_peak_matrix, regions, all_data = None):
        """ Runs predictions on a matrix of accessibility peaks, where columns are samples and
        rows are regions from regions_peak_file. rows in accessilibility_peak_matrix should matching

        Args:
            :param accessilibility_peak_matrix: numpy matrix of (samples by genomic regions)
            :param regions_peak_file: either narrowpeak or bed file containing regions to score, OR a pyranges object
                with columns [Chomosome, Start, End, idx]. Index matches each genomic region to a row in
                accessilibility_peak_matrix. In both cases, number of regions Should
                match rows in accessilibility_peak_matrix
            :param all_data: for testing. If none, generates a concatenated matrix of all data when called.

        Returns:
            3-dimensional numpy matrix of predictions: sized (samples by regions by ChIP-seq targets)
        """

        if type(regions) == str:
            regions_bed = bed2Pyranges(regions)
        elif type(regions) == pr.PyRanges:
            regions_bed = regions
        else:
            raise Exception("regions must be type scring or pr.Pyranges, but got type %s" % type(regions))

        joined = regions_bed.join(self.dataset.regions, how='left',suffix='_alldata').df

        idx = joined['idx_alldata']

        results = []

        # TODO 9/10/2020: should do something more efficiently than a for loop
        for sample_i in tqdm.tqdm(range(accessilibility_peak_matrix.shape[0])):
            # tuple of means and stds
            peaks_i = np.zeros((len(self.dataset.regions)))
            peaks_i[idx] = accessilibility_peak_matrix[sample_i, joined['idx']]

            means, _ = self.eval_vector(peaks_i, idx, samples = 1)

            # group means by joined['idx']
            results.append(means)

        # stack all samples along 0th axis
        tmp = np.stack(results)

        # get the index break for each region_bed region
        reduce_indices = joined.drop_duplicates('idx',keep='first').index.values

        # get the number of times there was a scored region for each region_bed region
        # used to calculate reduced means
        indices_counts = joined['idx'].value_counts(sort=False).values[:,None]

        # reduce means on middle axis
        final = np.add.reduceat(tmp, reduce_indices, axis = 1)/indices_counts

        # TODO 9/10/2020: code is currently scoring missing values and setting to nan.
        # You could be more efficient and remove these so you are not taking
        # the time to score garbage data.

        # fill missing indices with nans
        missing_indices = joined[joined['idx_alldata']==-1]['idx'].values
        final[:,missing_indices, :] = np.NAN

        return final


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
        similarity_peak_prs = [bed2Pyranges(f) for f in similarity_peak_files]
        peak_matrix = np.vstack([pyranges2Vector(f, self.dataset.regions)[0] for f in similarity_peak_prs])

        peak_vector_regions, all_peaks_regions = pyranges2Vector(bed2Pyranges(regions_peak_file), self.dataset.regions)

        print("finished loading peak file")

        # only select peaks to score
        idx = np.where(peak_vector_regions == True)[0]

        print("scoring %i regions" % idx.shape[0])

        if len(idx) == 0:
            raise ValueError("No positive peaks found in %s" % regions_peak_file)

        # tuple of means and stds
        means, stds = self.eval_vector(peak_matrix, idx)
        print("finished predictions...", means.shape)

        assert type(means) == type(stds), "Means and STDs variables not of the same type"
        if not isinstance(means, np.ndarray):
            means = means.numpy()
            stds = stds.numpy()


        means_df =  pd.DataFrame(data=means, columns=self.dataset.predict_targets)
        std_cols = list(map(lambda x: x + "_stds",self.dataset.predict_targets))
        stds_df =  pd.DataFrame(data=stds, columns=std_cols)

        # read in regions file and filter by indices that were scored
        p = self.dataset.regions.df
        p['idx']=p.index # keep original bed region ordering using idx column
        p.columns = ['Chromosome', 'Start','End','idx']
        prediction_positions = p[p['idx'].isin(idx)] # select regions that were scored
        # reset index to match predictions shape
        prediction_positions = prediction_positions.reset_index()
        prediction_positions['idx'] = prediction_positions.index
        prediction_positions = pd.concat([prediction_positions,means_df,stds_df],axis=1)
        prediction_positions_pr = pr.PyRanges(prediction_positions, int64=True).sort()

        original_file = bed2Pyranges(regions_peak_file)
        # left join
        joined = original_file.join(prediction_positions_pr, how='left')
        # turn into dataframe
        joined_df = joined.df
        joined_df = joined_df.drop(['index','Start_b','End_b','idx_b'],axis=1)
        # set missing values to 0
        num = joined_df._get_numeric_data()
        num[num < 0] = 0
        mean_results = joined_df.groupby('idx').mean()

        tmp = original_file.df
        tmp = tmp.set_index(tmp['idx']) # correctly set index to original ordering

        assert np.all(tmp.sort_index()['Start'].values == mean_results['Start'].values), "Error: genomic position start sites in mean results and original bed file do not match!"
        assert np.all(tmp.sort_index()['End'].values == mean_results['End'].values), "Error: genomic position end sites in mean results and original bed file do not match!"

        return pd.concat([tmp[['Chromosome']],mean_results], axis=1)

class VLP(VariationalPeakModel):
    def __init__(self,
             *args,
             **kwargs):
        '''
        Creates a new model with 4 layers with 100 unites each.
            To resume model training on an old model, call:

            .. code-block:: python

                model = VLP(checkpoint=path_to_saved_model)
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

            VariationalPeakModel.__init__(self, **metadata, **kwargs)
            file = h5py.File(os.path.join(kwargs["checkpoint"], "weights.h5"), 'r')

            # load model weights back in
            weights = []
            for i in range(len(file.keys())):
                weights.append(file['weight' + str(i)][:])
            self.model.set_weights(weights)
            file.close()

        else:
            VariationalPeakModel.__init__(self, *args, **kwargs)

    def create_model(self, **kwargs):
        '''
        Creates an Epitome model.
        '''
        cell_inputs = [tf.keras.layers.Input(shape=(self.num_inputs[i],))
                       for i in range(len(self.num_inputs))]

        # make a channel for each cell type
        cell_channels = []

        # TODO resize by max iterations. 5000 is an estimate for data size
        kl_divergence_function = (lambda q, p, _: tfp.distributions.kl_divergence(q, p) /
                            tf.cast(self.batch_size * 5000, dtype=tf.float32))
        for i in range(len(self.num_inputs)):
            # make input layer for cell
            last = cell_inputs[i]
            for j in range(self.layers):
                num_units = int(self.num_inputs[i]/(2 * (j+1)))
                d = tfp.layers.DenseFlipout(num_units,
                                                kernel_divergence_fn=kl_divergence_function,
                                                activation = self.activation)(last)
                last = d
            cell_channels.append(last)


        # merge together all cell channels
        if (len(cell_channels) > 1):
            last = tf.keras.layers.concatenate(cell_channels)
        else:
            last = cell_channels[0]

        outputs = tfp.layers.DenseFlipout(self.num_outputs,
                                        kernel_divergence_fn=kl_divergence_function,
                                        activity_regularizer=tf.keras.regularizers.l1_l2(self.l1, self.l2),
                                        name="output_layer")(last)

        model = tf.keras.models.Model(inputs=cell_inputs, outputs=outputs)
        return model
