"""
Functions and classes for model specifications.
"""

import sklearn.metrics


import tensorflow as tf
from .constants import *
from .functions import *
from .generators import *
import numpy as np

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
        :param generator
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

        """
        
        self.graph = tf.Graph()
    
        with self.graph.as_default() as graph:
            
            tf.compat.v1.logging.set_verbosity(tf.logging.INFO)
            
                
            assert (set(test_celltypes) < set(list(cellmap))), \
                    "test_celltypes %s must be subsets of available cell types %s" % (str(test_celltypes), str(list(cellmap)))
                
                
            # get evaluation cell types by removing any cell types that would be used in test
            self.eval_cell_types = list(cellmap).copy()
            self.test_celltypes = test_celltypes
            
            [self.eval_cell_types.remove(test_cell) for test_cell in self.test_celltypes]
            print("eval cell types", self.eval_cell_types)

            output_shape, train_iter = generator_to_one_shot_iterator(load_data(data[Dataset.TRAIN],  
                                                    self.eval_cell_types,
                                                    self.eval_cell_types,
                                                    matrix,
                                                    assaymap,
                                                    cellmap,
                                                    radii = radii, mode = Dataset.TRAIN),
                                                    batch_size, shuffle_size, prefetch_size)

            _,            valid_iter = generator_to_one_shot_iterator(load_data(data[Dataset.VALID], 
                                                    self.eval_cell_types,
                                                    self.eval_cell_types,
                                                    matrix,
                                                    assaymap,
                                                    cellmap,
                                                    radii = radii, mode = Dataset.VALID), 
                                                    batch_size, 1, prefetch_size)

            # can be empty if len(test_celltypes) == 0
            _,            test_iter = generator_to_one_shot_iterator(load_data(data[Dataset.TEST], 
                                                   self.test_celltypes, 
                                                   self.eval_cell_types,
                                                   matrix,
                                                   assaymap,
                                                   cellmap,
                                                   radii = radii, mode = Dataset.TEST),
                                                       batch_size, 1, prefetch_size)

            
            self.train_handle = train_iter.string_handle()
            self.valid_handle = valid_iter.string_handle()
            self.test_handle = test_iter.string_handle()
            
            self.handle = tf.placeholder(tf.string, shape=[])
            
            iterator = tf.data.Iterator.from_string_handle(
                self.handle, train_iter.output_types, train_iter.output_shapes)
            
            # self.x = predictions, self.y=labels, self.z = missing labels for this record (cell type specific)
            self.x, self.y, self.z = iterator.get_next()

            # Set session and graph. to interact with model outside functions,
            # you will have to use "with model.graph.as_default():" (ie for creating generators and such)
            config = tf.ConfigProto(log_device_placement=False)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(graph=graph, config=config)
            self.graph = graph

            self.num_outputs = output_shape[0]
            self.l1, self.l2 = l1, l2
            self.default_lr = lr
            self.lr = tf.placeholder(tf.float32)
            self.batch_size = batch_size
            self.prefetch_size = prefetch_size
            self.shuffle_size = shuffle_size

            # set self
            self.radii = radii
            self.debug = debug
            self.assaymap = assaymap
            self.test_celltypes = test_celltypes
            self.matrix = matrix
            self.assaymap= assaymap 
            self.cellmap = cellmap
            self.data = data
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.logits = self.body_fn()
            self.predictions = tf.sigmoid(self.logits)
            self.loss = self.loss_fn()
            self.min = self.minimizer_fn()
            
            self.closed = False
        
            
    def save(self, checkpoint_path):
        save_path = self.saver.save(self.sess, checkpoint_path)
        tf.logging.info("Model saved in path: %s" % save_path)
        
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
        
        fileObject = open(checkpoint_path + ".model_params.pickle",'wb')
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

    def restore(self, checkpoint_path):
        # need to kickstart train to create saver variable
        self.train(0)
        self.train(0, checkpoint_path = checkpoint_path)
        
    def body_fn(self):
        raise NotImplementedError()
    
    def loss_fn(self):
        # weighted sum of cross entropy for non 0 weights
        # Reduction method = Reduction.SUM_BY_NONZERO_WEIGHTS
        individual_losses = tf.losses.sigmoid_cross_entropy(self.y,  # TODO AM 5/23/2019 right dimensions
                                                            self.logits[:, Features.FEATURE_IDX.value, :], 
                                                            weights = self.z,
                                                            reduction = tf.losses.Reduction.NONE)
        
        # TODO AM 5/17/2019 what about missing labels (weights)?
        individual_losses = tf.math.reduce_mean(individual_losses, axis = 0)
        
        return (tf.losses.sigmoid_cross_entropy(self.y, self.logits[:, Features.FEATURE_IDX.value, :], weights = self.z), 
                individual_losses)
    
    def minimizer_fn(self):
        self.opt = tf.train.AdamOptimizer(self.lr)
        return self.opt.minimize(self.loss[0], self.global_step)
    
    def resample_train(self, individual_losses):
        """
        Resamples training dataset weighted by losses. Factors with higher loss will
        be upsampled more. Resets the train handle.
        
        :param individual_losses: tensor of losses for each factor. Shape is 1-dimensional array of number of factors
        """
        new_train_data_indices = indices_for_weighted_resample(self.data[Dataset.TRAIN], 1000, 
                                                                     self.matrix, 
                                                                     self.cellmap, 
                                                                     self.assaymap, 
                                                                     weights = individual_losses)

        # reset train handle
        output_shape, train_iter = generator_to_one_shot_iterator(load_data(self.data[Dataset.TRAIN],  
                                        self.eval_cell_types,
                                        self.eval_cell_types,
                                        self.matrix,
                                        self.assaymap,
                                        self.cellmap,
                                        indices = new_train_data_indices,
                                        radii = self.radii, mode = Dataset.TRAIN),
                                        self.batch_size, 1, self.prefetch_size)
        
        self.train_handle = train_iter.string_handle()
        self.train_handle = self.sess.run(self.train_handle)
        
        
    def close(self):
        if not self.closed:
            self.sess.close()
        self.closed = True
        
    def train(self, num_steps, lr=None, checkpoint_path = None):
        assert not self.closed
        with self.graph.as_default():
            if lr == None:
                lr = self.default_lr
            try:
                if (checkpoint_path != None):
                    self.saver.restore(self.sess, checkpoint_path)
                else:
                    self.sess.run(self.global_step)
            except:
                tf.logging.info("Initializing variables")
                self.sess.run(tf.global_variables_initializer())
                self.saver = tf.train.Saver()  # define a saver for saving and restoring
                self.train_handle = self.sess.run(self.train_handle)
                self.valid_handle = self.sess.run(self.valid_handle)
                self.test_handle = self.sess.run(self.test_handle)

            max_steps = self.sess.run(self.global_step) + num_steps

            tf.logging.info("Starting Training")

            while self.sess.run(self.global_step) < max_steps:
                _, loss = self.sess.run([self.min, self.loss], {self.handle: self.train_handle, self.lr: lr})
                step = self.sess.run(self.global_step)
                if step % 1000 == 0:
                    tf.logging.info(str(step) + " " + str(loss[0]))
                    # weighted resample based on losses
#                     self.resample_train(loss[1])
                    if (self.debug):
                        tf.logging.info("On validation")
                        _, _, _, _, _ = self.test(40000, log=False)
                        tf.logging.info("")
             
    def test(self, num_samples, mode = Dataset.VALID, log=False, iterator_handle=None):
        """
        Tests model on valid and test dataset handlers.
        """

        if (mode == Dataset.VALID):
            handle = self.valid_handle # for standard validation of validation cell types
            
        elif (mode == Dataset.TEST and len(self.test_celltypes) > 0):
            handle = self.test_handle # for standard validation of validation cell types        
        else:
            raise Exception("No data exists for %s. Use function test_from_generator() if you want to create a new iterator." % (mode))
            
        return self.run_predictions(num_samples, handle, log)             

        
    def test_from_generator(self, num_samples, iter_, log=False):
        """
        Runs test given a specified data generator 
        :param num_samples: number of samples to test
        :param iter_: output of generator_to_one_shot_iterator(), one shot iterator of records
        :param cell_type: cell type to test on. Used to generate holdout indices.
        
        :return predictions
        """
        handle = iter_.string_handle()
        iter_handle = self.sess.run(handle)        
        return self.run_predictions(num_samples, iter_handle, log)
    
    def eval_vector(self, data, vector, indices):
        """
        Evaluates a new cell type based on its chromatin (DNase or ATAC-seq) vector. len(vector) should equal
        the data.shape[1]
        :param data: data to build features from 
        :param vector: vector of 0s/1s of binding sites TODO AM 4/3/2019: try peak strength instead of 0s/1s
        :param indices: indices of vector to actually score. You need all of the locations for the generator.

        :return predictions for all factors
        """
        
        _,  iter_ = generator_to_one_shot_iterator(load_data(data, 
                 self.test_celltypes,   # used for labels. Should be all for train/eval and subset for test
                 self.eval_cell_types,   # used for rotating features. Should be all - test for train/eval
                 self.matrix,
                 self.assaymap,
                 self.cellmap,
                 radii = self.radii,
                 mode = Dataset.RUNTIME,
                 dnase_vector = vector, indices = indices), self.batch_size, 1, self.prefetch_size)
            
        handle = iter_.string_handle()
        iter_handle = self.sess.run(handle)

        num_samples = len(indices)
        
        preds, _, _, _, _ = self.run_predictions(num_samples, iter_handle, False, calculate_metrics = False)    
        
        return preds

    def run_predictions(self, num_samples, iter_handle, log, calculate_metrics = True):
        """
        Runs predictions on num_samples records
        :param num_samples: number of samples to test
        :param iter_: output of self.sess.run(generator_to_one_shot_iterator()), handle to one shot iterator of records
        :param log: if true, logs individual factor accuracies
        
        :return preds, truth, assay_dict, auROC, auPRC, False
            preds = predictions, 
            truth = actual values, 
            assay_dict = if log=True, holds predictions for individual factors
            auROC = average macro area under ROC for all factors with truth values
            auPRC = average area under PRC for all factors with truth values
        """
        
        inv_assaymap = {v: k for k, v in self.assaymap.items()}
        
        assert not self.closed
        with self.graph.as_default():
            vals = []
            for i in range(int(num_samples / self.batch_size)+1): # +1 to account for remaining samples % batch_size
                vals.append(
                    self.sess.run([self.predictions, self.x, self.y, self.z],
                             {self.handle: iter_handle})
                )
                
            preds = np.concatenate([v[0] for v in vals])   
            preds = preds[:,Features.FEATURE_IDX.value,:] # get feature row
            
            # do not continue to calculate metrics. Just return predictions
            if (not calculate_metrics):
                return preds, None, None, None, None
        
            truth = np.concatenate([v[2] for v in vals])
            sample_weight  = np.concatenate([v[3] for v in vals])
            
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
                
                for j in range(preds.shape[1]):
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
                    
                    
                    assay_dict[assay]= {"AUC": roc_score, "auPRC": pr_score, "GINI": gini_score }

                        
                
                auROC = np.nanmean(auROC_vec)
                auPRC = np.nanmean(auPRC_vec)

                tf.logging.info("macro auROC:     " + str(auROC))
                tf.logging.info("auPRC:     " + str(auPRC))
                tf.logging.info("GINI:     " + str(np.nanmean(GINI_vec)))
            except ValueError as v:
                auROC = None
                auPRC = None
                tf.logging.info("Failed to calculate metrics")

            return preds, truth, assay_dict, auROC, auPRC
        
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
            fileObject = open(kwargs["checkpoint"] + ".model_params.pickle" ,'rb')
            metadata = pickle.load(fileObject)
            PeakModel.__init__(self, kwargs["data"], **metadata)
        else: 
            PeakModel.__init__(self, *args, **kwargs)
            
    def body_fn(self):
        model = self.x

        if not isinstance(self.num_units, collections.Iterable):
            self.num_units = [self.num_units] * self.layers
        for i in range(self.layers):
            model = tf.layers.dense(model, self.num_units[i], self.activation)
        
        return tf.layers.dense(model, self.num_outputs, kernel_regularizer=tf.contrib.layers.l1_l2_regularizer(self.l1, self.l2))

