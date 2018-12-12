"""
Functions and classes for model specifications.
"""


################### Simple DNase peak based distance model ############

class PeakModel():
    def __init__(self,
                 train_data,
                 valid_data,
                 test_data,
                 validation_celltypes,
                 test_celltypes,
                 generator,
                 matrix,
                 assaymap,
                 cellmap,
                 label_assays = None, # what assays are you trying to predict in the label space?  If none, then all!  
                 batch_size=64,
                 shuffle_size=10000,
                 prefetch_size=10,
                 all_eval_cell_types=None, # Here, you can specify extra cell types you do not want in the train set (metalearn)
                 l1=0.,
                 l2=0.,
                 lr=1e-3,
                 radii=[1,3,10,30]):
        
        self.graph = tf.Graph()
    
        with self.graph.as_default() as graph:
            
            tf.logging.set_verbosity(tf.logging.INFO)
            
            if (all_eval_cell_types == None):
                self.all_eval_cell_types = validation_celltypes + test_celltypes
            else:
                self.all_eval_cell_types = all_eval_cell_types
                
            assert (set(validation_celltypes) < set(self.all_eval_cell_types)) & (set(test_celltypes) < set(self.all_eval_cell_types)), \
                    "%s and %s must be subsets of the specified eval cell types (%s)" % (validation_celltypes, test_celltypes, all_eval_cell_types)
            
            self.label_assays = label_assays
            
            # if label_assays is not specified, set it to the 
            # assays in the training set
            if (self.label_assays == None):
                self.label_assays = list(assaymap)
            
                
            # make datasets
            output_shape, train_iter = generator_to_one_shot_iterator(make_dataset(train_data,  
                                                    validation_celltypes, 
                                                    self.all_eval_cell_types,
                                                    generator, 
                                                    matrix,
                                                    assaymap,
                                                    cellmap,
                                                    label_assays = self.label_assays,
                                                    radii = radii), batch_size, shuffle_size, prefetch_size)
            shape,            valid_iter = generator_to_one_shot_iterator(make_dataset(valid_data, 
                                                        validation_celltypes, 
                                                        self.all_eval_cell_types,
                                                        generator, 
                                                        matrix,
                                                        assaymap,
                                                        cellmap,
                                                        label_assays = self.label_assays,
                                                        radii = radii), batch_size, 1, prefetch_size)
            _,            test_iter = generator_to_one_shot_iterator(make_dataset(test_data, 
                                                   test_celltypes, 
                                                   self.all_eval_cell_types,
                                                   generator, 
                                                   matrix,
                                                   assaymap,
                                                   cellmap,
                                                   label_assays = self.label_assays,
                                                   radii = radii), batch_size, 1, prefetch_size)

            self.train_handle = train_iter.string_handle()
            self.valid_handle = valid_iter.string_handle()
            self.test_handle = test_iter.string_handle()
            
            self.handle = tf.placeholder(tf.string, shape=[])
            iterator = tf.data.Iterator.from_string_handle(
                self.handle, train_iter.output_types, train_iter.output_shapes)
            self.x, self.y = iterator.get_next()
            
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

            self.label_matrix, self.label_cellmap, self.label_assaymap = get_assays_from_feature_file(feature_path='../data/feature_name', 
                                 eligible_assays = self.label_assays, 
                                 eligible_cells = list(cellmap), min_assays = 0)
            
            assert self.label_cellmap == self.cellmap, "cellmaps for label space and feature space are not the same"
        
        
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.logits = self.body_fn()
            self.predictions = tf.sigmoid(self.logits)
            self.loss = self.loss_fn()
            self.min = self.minimizer_fn()
            
            self.closed = False
        
            
    def save(self, checkpoint_path):
        save_path = self.saver.save(self.sess, checkpoint_path)
        tf.logging.info("Model saved in path: %s" % save_path)
        
        
    def restore(self, checkpoint_path):
        # need to kickstart train to create saver variable
        self.train(0)
        self.train(0, checkpoint_path = checkpoint_path)
        
    def body_fn(self):
        raise NotImplementedError()
    
    def loss_fn(self):
        return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(self.y, self.logits, 50))
    
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
                                       label_assays = self.label_assays,
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
            validation_holdout_indices = get_missing_indices_for_cell(self.label_matrix, 
                                                                      self.cellmap, 
                                                                      self.validation_celltypes[0]) - 1
            if (len(self.validation_celltypes) > 1):
                assert len(validation_holdout_indices) == 0, \
                    """
                    Error: validation_holdout_indices has > 0 elements, and there are multiple validation_celltypes.
                    This will cause incorrect accuracy reportings.
                    """
            
        elif (mode == Dataset.TEST):
            handle = self.test_handle # for standard validation of validation cell types
            validation_holdout_indices = get_missing_indices_for_cell(self.label_matrix, 
                                                                      self.cellmap, 
                                                                      self.test_celltypes[0]) - 1 # shift b/c dnase was removed
        else:
            raise Exception("No data handler exists for %s. Use function test_from_generator() if you want to create a new iterator." % (mode))
            
        return self.run_predictions(num_samples, handle, validation_holdout_indices, log)             

        
    def test_from_generator(self, num_samples, iter_, validation_holdout_indices, log=False):
        """
        Runs test given a specified data generator 
        :param num_samples: number of samples to test
        :param iter_: output of generator_to_one_shot_iterator(), one shot  iterator of records
        :param cell_type: cell type to test on. Used to generate holdout indices.
        
        :return predictions
        """
        handle = iter_.string_handle()
        iter_handle = self.sess.run(handle)        
        return self.run_predictions(num_samples, iter_handle, validation_holdout_indices, log)
    
    # TODO am TEST
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
                               label_assays = self.label_assays,
                               radii = radii,
                               dnase_vector = vector)

        
        handle = iter_.string_handle()
        iter_handle = self.sess.run(handle)
        validation_holdout_indices = get_missing_indices_for_cell(self.label_matrix, 
                                                                  self.cellmap, cell_type) - 1 # shift b/c dnase was removed

        num_samples = data['x'].shape[1]
        
        return self.run_predictions(num_samples, iter_handle, validation_holdout_indices, log)
        
    def run_predictions(self, num_samples, iter_handle, validation_holdout_indices, log):
        
        inv_assaymap = {v: k for k, v in self.label_assaymap.items()}
        
        assert not self.closed
        with self.graph.as_default():
            vals = []
            for i in range(int(num_samples / self.batch_size)):
                vals.append(
                    self.sess.run([self.predictions, self.x, self.y],
                             {self.handle: iter_handle})
                )
            preds = np.concatenate([v[0] for v in vals])            
            truth = np.concatenate([v[2] for v in vals])

            # remove missing indices for computing macro/micro AUC
            preds_r = np.delete(preds, validation_holdout_indices, axis=1)
            truth_r = np.delete(truth, validation_holdout_indices, axis=1)
            
            try:
                macroAUC = sklearn.metrics.roc_auc_score(truth_r, preds_r, average='macro')
                microAUC = sklearn.metrics.roc_auc_score(truth_r, preds_r, average='micro')
                tf.logging.info("Our macro AUC:     " + str(macroAUC))
                tf.logging.info("Our micro AUC:     " + str(microAUC))
            except ValueError:
                macroAUC = None
                microAUC = None
                tf.logging.info("Failed to calculate macro AUC")
                tf.logging.info("Failed to calculate micro AUC")
                
            if log:
                j=0

                for i in range(self.label_matrix.shape[1]): # eval on all assays except DNase and assays that are missing 
                    assay = inv_assaymap[i]

                    if (i not in validation_holdout_indices+1 and i != 0):
                        try:
                            str_ = "%s: %i, %s, %f" % (str(datetime.datetime.now()), i, assay, sklearn.metrics.roc_auc_score(truth_r[:,j], preds_r[:,j], average='macro'))
                        except ValueError:
                            str_ = "%s: %i, %s, CANT CALCULATE" % (str(datetime.datetime.now()), i, assay)
                        j = j + 1
                    else:
                        print(str(datetime.datetime.now()), i, assay)
                        str_ = "%s: %i, %s, NaN" % (str(datetime.datetime.now()), i, assay)

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

        