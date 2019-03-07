"""
Functions and classes for model specifications.
"""
# TODOS
# - AM 3/7/2019: Currently, there are -1's in the feature vectors from the generator
# that indicating missing labels. We are not doing anything smart here. Is there something better 
# we can do?

# Calculating PR scores
from sklearn.metrics import average_precision_score

################### Simple DNase peak based distance model ############

"""
Functions and classes for model specifications.
"""

# Calculating PR scores
from sklearn.metrics import average_precision_score

################### Simple DNase peak based distance model ############

class PeakModel():
    def __init__(self,
                 train_data,
                 valid_data,
                 test_data,
                 test_celltypes,
                 generator,
                 matrix,
                 assaymap,
                 cellmap,  
                 batch_size=64,
                 shuffle_size=10000,
                 prefetch_size=10,
                 all_eval_cell_types=None, # Here, you can specify extra cell types you do not want in the train set (metalearn)
                 l1=0.,
                 l2=0.,
                 lr=1e-3,
                 radii=[1,3,10,30]):
        
        """
        Peak Model

        :param train_data
        :param valid_data
        :param test_celltypes
        :param generator
        :param matrix
        :param assaymap
        :param cellmap
        :param batch_size
        :param shuffle_size
        :param prefetch_size
        :param all_eval_cell_types
        :param l1
        :param l2
        :param lr
        :param radii

        """
        
        self.graph = tf.Graph()
    
        with self.graph.as_default() as graph:
            
            tf.logging.set_verbosity(tf.logging.INFO)
            
                
            assert (set(test_celltypes) < set(list(cellmap))), \
                    "test_celltypes %s must be subsets of available cell types" % (test_celltypes, list(celltypes))
                
                
            # get evaluation cell types by removing any cell types that would be used in test
            self.eval_cell_types = list(cellmap).copy()
            [self.eval_cell_types.remove(test_cell) for test_cell in test_celltypes]
            print("eval cell types", self.eval_cell_types)

            # make datasets
            output_shape, train_iter = generator_to_one_shot_iterator(make_dataset(train_data,  
                                                    self.eval_cell_types,
                                                    self.eval_cell_types,
                                                    generator, 
                                                    matrix,
                                                    assaymap,
                                                    cellmap,
                                                    radii = radii, mode = Dataset.TRAIN),
                                                    batch_size, shuffle_size, prefetch_size)
                                                                      
            _,            valid_iter = generator_to_one_shot_iterator(make_dataset(valid_data, 
                                                    self.eval_cell_types,
                                                    self.eval_cell_types,
                                                    generator, 
                                                    matrix,
                                                    assaymap,
                                                    cellmap,
                                                    radii = radii, mode = Dataset.VALID), 
                                                    batch_size, 1, prefetch_size)
            
            _,            test_iter = generator_to_one_shot_iterator(make_dataset(test_data, 
                                                   test_celltypes, 
                                                   self.eval_cell_types,
                                                   generator, 
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
            
            self.sess = tf.InteractiveSession(graph=graph)

            self.num_outputs = output_shape[0]
            self.l1, self.l2 = l1, l2
            self.default_lr = lr
            self.lr = tf.placeholder(tf.float32)
            self.batch_size = batch_size
            self.prefetch_size = prefetch_size

            # set self
            self.assaymap = assaymap
            self.all_eval_cell_types = validation_celltypes + test_celltypes
            self.validation_celltypes = validation_celltypes
            self.test_celltypes = test_celltypes
            self.generator = generator
            self.matrix = matrix
            self.assaymap= assaymap 
            self.cellmap = cellmap
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.logits = self.body_fn()
            self.predictions = tf.sigmoid(self.logits)
            self.loss = self.loss_fn()
            self.min = self.minimizer_fn()
            
            self.closed = False
        
            
    def save(self, checkpoint_path):
        save_path = self.saver.save(self.sess, checkpoint_path)
        tf.logging.info("Model saved in path: %s" % save_path)
        
        
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
        cross_entropy = tf.nn.weighted_cross_entropy_with_logits(self.y, self.logits, 1)
        # mask cross entropy by weights z and take mean
        return tf.reduce_mean(tf.boolean_mask(cross_entropy, self.z) )
    
    def minimizer_fn(self):
        self.opt = tf.train.AdamOptimizer(self.lr)
        return self.opt.minimize(self.loss, self.global_step)
    
    def close():
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
                    tf.logging.info(str(step) + " " + str(loss))
                    tf.logging.info("On validation")
                    _, _, _, _, stop = self.test(40000, log=True)
                    if stop: break
                    tf.logging.info("")
             
    # TODO TEST
    def eval_vector(self, data, vector, log=False):
        
        _, iter_ = make_dataset(data, 
                                       self.test_celltypes, 
                                       self.all_eval_cell_types,
                                       gen_from_chromatin_vector, 
                                       self.matrix,
                                       self.assaymap,
                                       self.cellmap,
                                       self.batch_size, 
                                       1           , 
                                       self.prefetch_size, 
                                       radii = radii,
                                       dnase_vector = vector)

            
        handle = iter_.string_handle()
        iter_handle = self.sess.run(handle)
        validation_holdout_indices = []

        num_samples = data['y'].shape[1]
        
        predictions, _, _, _, _ = self.run_predictions(num_samples, iter_handle, validation_holdout_indices, log)
        return predictions                       
                               
    def test(self, num_samples, mode = Dataset.VALID, log=False, iterator_handle=None):
        """
        Tests model on valid and test dataset handlers.
        """
        
        if (mode == Dataset.VALID):
            handle = self.valid_handle # for standard validation of validation cell types
            
        elif (mode == Dataset.TEST):
            handle = self.test_handle # for standard validation of validation cell types        
            
        else:
            raise Exception("No data handler exists for %s. Use function test_from_generator() if you want to create a new iterator." % (mode))
            
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
    
    # TODO AM TEST
    def test_vector(self, WHICH_DATASET, dnase_vector, cell_type, log=False):

        # make dataset from DNase vector
        if (WHICH_DATASET == Dataset.TRAIN):
            data = train_data
        elif (WHICH_DATASET == Dataset.VALID):
            data = valid_data
        elif (WHICH_DATASET == Dataset.TEST):
            data = test_data

        _, iter_ = make_dataset(data, 
                               celltype, 
                               self.all_eval_cell_types,
                               gen_from_chromatin_vector, 
                               self.matrix,
                               self.assaymap,
                               self.cellmap,
                               self.batch_size, 
                               1           , 
                               self.prefetch_size, 
                               radii = radii,
                               dnase_vector = vector)

        
        handle = iter_.string_handle()
        iter_handle = self.sess.run(handle)

        num_samples = data['x'].shape[1]
        
        return self.run_predictions(num_samples, iter_handle, log)
        
    def run_predictions(self, num_samples, iter_handle, log):
        
        inv_assaymap = {v: k for k, v in self.assaymap.items()}
        
        assert not self.closed
        with self.graph.as_default():
            vals = []
            for i in range(int(num_samples / self.batch_size)):
                vals.append(
                    self.sess.run([self.predictions, self.x, self.y, self.z],
                             {self.handle: iter_handle})
                )
            preds = np.concatenate([v[0] for v in vals])            
            truth = np.concatenate([v[2] for v in vals])
            sample_weight  = np.concatenate([v[3] for v in vals])

            assert(preds.shape == sample_weight.shape)
            
            # TODO AM 3/6/2019 shrink down accuracy calculations
            try:
                # Mean results because sample_weight mask can only work on 1 row at a time.
                # If a given assay is not available for evaluation, sample_weights will all be 0 
                # and the resulting roc_auc_score will be NaN.
                macroAUC_vec = []
                microAUC_vec = []
                
                # try/accept for cases with only one class (throws ValueError)
                for k in range(preds.shape[1]):
                    try:
                        macroAUC_vec.append(sklearn.metrics.roc_auc_score(truth[:,k], preds[:,k], 
                                                          average='macro', 
                                                          sample_weight = sample_weight[:,k]))
                    except ValueError:
                        pass
                    
                    try:
                        microAUC_vec.append(sklearn.metrics.roc_auc_score(truth[:,k], preds[:,k], 
                                                          average='micro', 
                                                          sample_weight = sample_weight[:,k]))
                    except ValueError:
                        pass

                
                macroAUC = np.nanmean(macroAUC_vec)
                microAUC = np.nanmean(microAUC_vec)

                tf.logging.info("Our macro AUC:     " + str(macroAUC))
                tf.logging.info("Our micro AUC:     " + str(microAUC))
            except ValueError as v:
                print(v.args)
                macroAUC = None
                microAUC = None
                tf.logging.info("Failed to calculate macro AUC")
                tf.logging.info("Failed to calculate micro AUC")
                
            if log:
                j=0
                

                for j in range(truth.shape[1]): # eval on all assays except DNase and assays that are missing 
                    assay = inv_assaymap[j+1]

                    try:
                        auc_score =  sklearn.metrics.roc_auc_score(truth[:,j], preds[:,j], 
                                                                   sample_weight = sample_weight[:, j], 
                                                                   average='macro')
                        
                        pr_score  =  average_precision_score(truth[:,j], preds[:,j], 
                                                             sample_weight = sample_weight[:, j])
                        
                        gini_score = self.gini_normalized(truth[:,j], preds[:,j], 
                                                          sample_weight = sample_weight[:, j])

                        str_ = "%s:\tAUC:%.3f\tavPR:%.3f\tGINI:%.3f" % (assay, auc_score, pr_score, gini_score)
                    except ValueError:
                        tf.logging.warn("unable to calculate metrics for assay %s" % assay)
                        str_ = "%s:\tAUC:NaN\tavPR:NaN\tGINI:NaN" % (assay)
                    j = j + 1

                    tf.logging.info(str_)

            return preds, truth, microAUC, macroAUC, False


class MLP(PeakModel):
    def __init__(self,
             layers,
             num_units,
             activation,
             *args,
             **kwargs):

        self.layers = layers
        self.num_units = num_units
        self.activation = activation
        self.radii = kwargs["radii"]

        PeakModel.__init__(self, *args, **kwargs)
            
    def body_fn(self):
        model = self.x
        
        if not isinstance(self.num_units, collections.Iterable):
            self.num_units = [self.num_units] * self.layers
            
        for i in range(self.layers):
            model = tf.layers.dense(model, self.num_units[i], self.activation)
        
        return tf.layers.dense(model, self.num_outputs, kernel_regularizer=tf.contrib.layers.l1_l2_regularizer(self.l1, self.l2))

        