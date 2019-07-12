"""
Functions and classes for model specifications.
"""

import sklearn.metrics


import tensorflow as tf
from .constants import *
from .functions import *
from .generators import *
import numpy as np

# for saving model
import pickle
from operator import itemgetter

# disable sklearn warnings when training
import warnings
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

################### Simple DNase peak based distance model ############

class PeakModel():
    def __init__(self,
                 data,
                 test_celltypes,
                 matrix,
                 assaymap,
                 cellmap,  
                 debug = False,
                 batch_size=64,
                 shuffle_size=10,
                 prefetch_size=10,
                 l1=0.,
                 l2=0.,
                 lr=1e-3,
                 radii=[1,3,10,30], 
                 train_indices = None):
        
        """
        Peak Model
        :param data: either a path to TF records OR a dictionary of TRAIN, VALID, and TEST data
        :param test_celltypes
        :param matrix
        :param assaymap
        :param cellmap
        :param debug: used to print out intermediate validation values
        :param batch_size
        :param shuffle_size
        :param prefetch_size
        :param l1
        :param l2
        :param lr
        :param radii
        :param train_indices: option numpy array of indices to train from data[Dataset.TRAIN]
        """
        
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

        assert (set(test_celltypes) < set(list(cellmap))), \
                "test_celltypes %s must be subsets of available cell types %s" % (str(test_celltypes), str(list(cellmap)))

        # get evaluation cell types by removing any cell types that would be used in test
        self.eval_cell_types = list(cellmap).copy()
        self.test_celltypes = test_celltypes
        [self.eval_cell_types.remove(test_cell) for test_cell in self.test_celltypes]
        print("eval cell types", self.eval_cell_types)

	assert (len(self.eval_cell_types) >= 2 ), \
		"there must be more than one eval_cell_type {} for feature rotation".format(self.eval_cell_types)


        self.output_shape, self.train_iter = generator_to_tf_dataset(load_data(data[Dataset.TRAIN],  
                                                self.eval_cell_types,
                                                self.eval_cell_types,
                                                matrix,
                                                assaymap,
                                                cellmap,
                                                radii = radii, mode = Dataset.TRAIN),
                                                batch_size, shuffle_size, prefetch_size)

        _,            self.valid_iter = generator_to_tf_dataset(load_data(data[Dataset.VALID], 
                                                self.eval_cell_types,
                                                self.eval_cell_types,
                                                matrix,
                                                assaymap,
                                                cellmap,
                                                radii = radii, mode = Dataset.VALID), 
                                                batch_size, 1, prefetch_size)

        # can be empty if len(test_celltypes) == 0
        _,            self.test_iter = generator_to_tf_dataset(load_data(data[Dataset.TEST], 
                                               self.test_celltypes, 
                                               self.eval_cell_types,
                                               matrix,
                                               assaymap,
                                               cellmap,
                                               radii = radii, mode = Dataset.TEST),
                                               batch_size, 1, prefetch_size)

        self.num_outputs = self.output_shape[0]
        self.l1, self.l2 = l1, l2
        self.lr = lr
        self.batch_size = batch_size
        self.prefetch_size = prefetch_size
        self.shuffle_size = shuffle_size
        self.optimizer =tf.keras.optimizers.Adam(lr=self.lr)

        # set self
        self.model = self.create_model()
        self.radii = radii
        self.debug = debug
        self.assaymap = assaymap
        self.test_celltypes = test_celltypes
        self.matrix = matrix
        self.assaymap= assaymap 
        self.cellmap = cellmap
        self.data = data
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

            
    def save(self, checkpoint_path):
        """
        Saves model.
        
        :param checkpoint_path: string file path to save model to. 
        """
        # save keras model
        self.model.save(checkpoint_path)
        
        # save model params to pickle file
        dict_ = {'test_celltypes':self.test_celltypes,
                         'matrix':self.matrix,
                         'assaymap':self.assaymap,
                         'cellmap':self.cellmap,  
                         'debug': self.debug,
                         'batch_size':self.batch_size,
                         'shuffle_size':self.shuffle_size,
                         'prefetch_size':self.prefetch_size,
                         'radii':self.radii}
        
        fileObject = open(os.path.join(checkpoint_path, "model_params.pickle"),'wb')
        pickle.dump(dict_,fileObject)   
        fileObject.close()
        
        
    def gini(self, actual, pred, sample_weight):                                                 
        df = sorted(zip(actual, pred), key=lambda x : (x[1], x[0]),  reverse=True)
        random = [float(i+1)/float(len(df)) for i in range(len(df))]                
        totalPos = np.sum([x[0] for x in df])           
        cumPosFound = np.cumsum([x[0] for x in df])                                     
        Lorentz = [float(x)/totalPos for x in cumPosFound]                          
        Gini = np.array([l - r for l, r in zip(Lorentz, random)])
        # mask Gini with weights
        Gini[np.where(sample_weight == 0)[0]] = 0
        return np.sum(Gini)    

    def gini_normalized(self, actual, pred, sample_weight = None):              
        normalized_gini = self.gini(actual, pred, sample_weight)/self.gini(actual, actual, sample_weight)      
        return normalized_gini       

    def body_fn(self):
        raise NotImplementedError()
        
    def g(self, p, a=1, B=0, y=1):
        """ Normalization Function. Normalizes loss w.r.t. label proportion.

        Constraints: 
         1. g(p) = 1 when p = 1
         2. g(p) = a * p^y + B, where a, y and B are hyperparameters
        """
        return a * tf.math.pow(p, y) + B
    
    def loss_fn(self, y_true, y_pred, weights):
        # weighted sum of cross entropy for non 0 weights
        # Reduction method = Reduction.SUM_BY_NONZERO_WEIGHTS
        loss = tf.compat.v1.losses.sigmoid_cross_entropy(y_true,  
                                                        y_pred[:, Features.FEATURE_IDX.value, :], 
                                                        weights = weights,
                                                        reduction = tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)

        C = (len(self.assaymap)-1)
        # p = tf.math.reduce_sum(weights, axis=1)/C # should be of dimension 1 by batch size
        p = 1.0 # this is just taking the mean loss, nothing special here.
        return self.g(p)/C * loss
        
    def train(self, num_steps, lr=None, checkpoint_path = None):
        if lr == None:
            lr = self.lr
            
        tf.compat.v1.logging.info("Starting Training")

        @tf.function
        def train_step(inputs, labels, weights):
            with tf.GradientTape() as tape:
                predictions = self.model(inputs, training=True)
                regularization_loss = tf.math.add_n(self.model.losses)
                pred_loss = self.loss_fn(labels, predictions, weights)
                total_loss = pred_loss + regularization_loss

            gradients = tape.gradient(total_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            return total_loss
        
        for step, (inputs, labels, weights) in enumerate(self.train_iter.take(num_steps)): 

            loss = train_step(inputs, labels, weights)

            if step % 1000 == 0:
                tf.compat.v1.logging.info(str(step) + " " + str(tf.reduce_mean(loss)))
                
                if (self.debug):
                    tf.compat.v1.logging.info("On validation")
                    _, _, _, _, _ = self.test(40000, log=False)
                    tf.compat.v1.logging.info("")

    def test(self, num_samples, mode = Dataset.VALID, log=False):
        """
        Tests model on valid and test dataset handlers.
        """

        if (mode == Dataset.VALID):
            handle = self.valid_iter # for standard validation of validation cell types
            
        elif (mode == Dataset.TEST and len(self.test_celltypes) > 0):
            handle = self.test_iter # for standard validation of validation cell types        
        else:
            raise Exception("No data exists for %s. Use function test_from_generator() if you want to create a new iterator." % (mode))
            
        return self.run_predictions(num_samples, handle, log)      
        
    def test_from_generator(self, num_samples, ds, log=False):
        """
        Runs test given a specified data generator 
        :param num_samples: number of samples to test
        :param ds: tensorflow dataset, created by dataset_to_tf_dataset
        :param cell_type: cell type to test on. Used to generate holdout indices.
        
        :return predictions
        """
        return self.run_predictions(num_samples, ds, log)
    
    def eval_vector(self, data, vector, indices):
        """
        Evaluates a new cell type based on its chromatin (DNase or ATAC-seq) vector. len(vector) should equal
        the data.shape[1]
        :param data: data to build features from 
        :param vector: vector of 0s/1s of binding sites TODO AM 4/3/2019: try peak strength instead of 0s/1s
        :param indices: indices of vector to actually score. You need all of the locations for the generator.

        :return predictions for all factors
        """
        
        _,  ds = generator_to_tf_dataset(load_data(data, 
                 self.test_celltypes,   # used for labels. Should be all for train/eval and subset for test
                 self.eval_cell_types,   # used for rotating features. Should be all - test for train/eval
                 self.matrix,
                 self.assaymap,
                 self.cellmap,
                 radii = self.radii,
                 mode = Dataset.RUNTIME,
                 dnase_vector = vector, indices = indices), self.batch_size, 1, self.prefetch_size)

        num_samples = len(indices)
        
        preds, _, _, _, _ = self.run_predictions(num_samples, ds, False, calculate_metrics = False)    
        
        return preds

    def run_predictions(self, num_samples, iter_, log, calculate_metrics = True):
        """
        Runs predictions on num_samples records
        :param num_samples: number of samples to test
        :param iter_: output of self.sess.run(generator_to_one_shot_iterator()), handle to one shot iterator of records
        :param log: if true, logs individual factor accuracies
        
        :return preds, truth, assay_dict, auROC, auPRC, False
            preds = predictions, 
            truth = actual values, 
            sample_weight: 0/1 weights on predictions.  
            assay_dict = if log=True, holds predictions for individual factors
            auROC = average macro area under ROC for all factors with truth values
            auPRC = average area under PRC for all factors with truth values
        """
        
        inv_assaymap = {v: k for k, v in self.assaymap.items()}
                
        # batches of predictions
        vals = []

        for inputs, labels, weights in iter_.take(int(num_samples / self.batch_size)+1): 
            vals.append([tf.sigmoid(self.model(inputs, training=False)), inputs, labels, weights])

            
        preds = np.concatenate([v[0] for v in vals])
        preds = preds[:,Features.FEATURE_IDX.value,:][:num_samples] # get feature row


        truth = np.concatenate([v[2] for v in vals])[:num_samples]
        sample_weight  = np.concatenate([v[3] for v in vals])[:num_samples]

        # do not continue to calculate metrics. Just return predictions and true values
        if (not calculate_metrics):
            return preds, truth, None, None, None

        assert(preds.shape == sample_weight.shape)

        try:
            # Mean results because sample_weight mask can only work on 1 row at a time.
            # If a given assay is not available for evaluation, sample_weights will all be 0 
            # and the resulting roc_auc_score will be NaN.
            auROC_vec = []
            auPRC_vec = []
            GINI_vec =  []


            # try/accept for cases with only one class (throws ValueError)
            assay_dict = {}

            for j in range(preds.shape[1]): # for all assays
                assay = inv_assaymap[j+1] 

                roc_score = np.NAN

                try:
                    roc_score = sklearn.metrics.roc_auc_score(truth[:,j], preds[:,j], 
                                                      average='macro', 
                                                      sample_weight = sample_weight[:,j])

                    auROC_vec.append(roc_score)

                except ValueError:
                    roc_score = np.NaN

                try:
                    pr_score = sklearn.metrics.average_precision_score(truth[:,j], preds[:,j], 
                                                             sample_weight = sample_weight[:, j])

                    auPRC_vec.append(pr_score)

                except ValueError:
                    pr_score = np.NaN

                try:
                    gini_score = self.gini_normalized(truth[:,j], preds[:,j], 
                                                      sample_weight = sample_weight[:, j])

                    GINI_vec.append(gini_score)

                except ValueError:
                    gini_score = np.NaN

                assay_dict[assay] = {"AUC": roc_score, "auPRC": pr_score, "GINI": gini_score }


            auROC = np.nanmean(auROC_vec)
            auPRC = np.nanmean(auPRC_vec)

            tf.compat.v1.logging.info("macro auROC:     " + str(auROC))
            tf.compat.v1.logging.info("auPRC:     " + str(auPRC))
            tf.compat.v1.logging.info("GINI:     " + str(np.nanmean(GINI_vec)))
        except ValueError as v:
            auROC = None
            auPRC = None
            tf.compat.v1.logging.info("Failed to calculate metrics")

        return preds, truth, sample_weight, assay_dict, auROC, auPRC
        
    def score_peak_file(self, peak_file):
    
        # get peak_vector, which is a vector matching train set. Some peaks will not overlap train set, 
        # and their indices are stored in missing_idx for future use
        peak_vector, all_peaks = bedFile2Vector(peak_file)
        print("finished loading peak file")

        # only select peaks to score
        idx = np.where(peak_vector == 1)[0]
        
        if len(idx) == 0:
            raise ValueError("No positive peaks found in %s" % peak_file)

        all_data = np.concatenate((self.data[Dataset.TRAIN], self.data[Dataset.VALID], self.data[Dataset.TEST]), axis=1)

        # takes about 1.5 minutes for 100,000 regions TODO AM 4/3/2019 speed up generator
        predictions = self.eval_vector(all_data, peak_vector, idx)
        print("finished predictions...", predictions.shape)


        # get number of factors to fill in if values are missing
        num_factors = predictions[0].shape[0]

        # map predictions with genomic position 
        liRegions = load_allpos_regions()
        prediction_positions = itemgetter(*idx)(liRegions)
        zipped = list(zip(prediction_positions, predictions))

        # for each all_peaks, if 1, reduce means for all overlapping peaks in positions
        # else, set to 0s

        def reduceMeans(peak):
            if (peak[1] == 1):
                # parse region

                # filter overlapping predictions for this peak and take mean      
                res = map(lambda k: k[1], filter(lambda x: peak[0].overlaps(x[0], 100), zipped))
                arr = np.concatenate(list(map(lambda x: np.matrix(x), res)), axis = 0)
                return(peak[0], np.mean(arr, axis = 0))
            else:
                return(peak[0], np.zeros(num_factors)) 

        grouped = list(map(lambda x: np.matrix(reduceMeans(x)[1]), all_peaks))

        final = np.concatenate(grouped, axis=0)

        df = pd.DataFrame(final, columns=list(self.assaymap)[1:])

        # load in peaks to get positions and could be called only once
        # TODO why are you reading this in twice?
        df_pos = pd.read_csv(peak_file, sep="\t", header = None)[[0,1,2]]
        final_df = pd.concat([df_pos, df], axis=1)

        return final_df
            

class MLP(PeakModel):
    def __init__(self,
             *args,
             **kwargs):

        """ To resume model training, call:
            model2 = MLP(data = data, checkpoint="/home/eecs/akmorrow/epitome/out/models/test_model")
        """
        self.layers = 4
        self.num_units = [100, 100, 100, 50]
        self.activation = tf.tanh
                          
        if "checkpoint" in kwargs.keys():
            fileObject = open(kwargs["checkpoint"] + "/model_params.pickle" ,'rb')
            metadata = pickle.load(fileObject)
            PeakModel.__init__(self, kwargs["data"], **metadata)
            self.model = tf.keras.models.load_model(kwargs["checkpoint"])
            
        else: 
            PeakModel.__init__(self, *args, **kwargs)
            
    def create_model(self):
        
        # this is called like model(data, training = True)
        model = tf.keras.Sequential()

        if not isinstance(self.num_units, collections.Iterable):
            self.num_units = [self.num_units] * self.layers
        for i in range(self.layers):
            model.add(tf.keras.layers.Dense(self.num_units[i], activation = self.activation))

        # output layer
        model.add(tf.keras.layers.Dense(self.num_outputs, kernel_regularizer=tf.keras.regularizers.l1_l2(self.l1, self.l2)))
        return model

