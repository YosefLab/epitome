from epitome.test import EpitomeTestCase
from epitome.constants import Dataset
import numpy as np
from epitome.models import VLP
import pytest

class ModelsTest(EpitomeTestCase):

	def __init__(self, *args, **kwargs):
		super(EpitomeTestCase, self).__init__(*args, **kwargs)
		self.model = self.makeSmallModel()
		self.validation_size = 10


	def test_train_model(self):
		train_iters = 2

		# create model and train
		self.model.train(1)
		results1 = self.model.test(self.validation_size)
		self.model.train(train_iters)
		results2 = self.model.test(self.validation_size)

		# Make sure predictions are not random
		# after first iterations
		assert(results1['preds_mean'].shape[0] == self.validation_size)
		assert(results2['preds_mean'][0] < results1['preds_mean'].shape[0])

    def test_train_early_stop_model(self):
        train_iters = 200

        # create model and train
        model = VLP(list(eligible_assays),
            test_celltypes = ['K562'],
            matrix = matrix, #np.ones((5,2)).astype(int),
            assaymap = assaymap,
            cellmap = cellmap,
            max_valid_records=10) 
        model.train(1)
        results1 = model.test(self.validation_size)
        model.train(train_iters)
        results2 = model.test(self.validation_size)

        # Make sure predictions are not random
        # after first iterations
        assert(results1['preds_mean'].shape[0] == self.validation_size)
        assert(results2['preds_mean'][0] < results1['preds_mean'].shape[0])

	def test_test_model(self):

		# make sure can run in test mode
		results = self.model.test(self.validation_size, mode=Dataset.TEST)
		assert(results['preds_mean'].shape[0] == self.validation_size)

	def test_eval_vector(self):

		# should be able to evaluate on a dnase vector
		similarity_matrix = np.ones(self.model.data[Dataset.TRAIN].shape[1])[None,:]
		results = self.model.eval_vector(self.model.data[Dataset.TRAIN], similarity_matrix, np.arange(0,20))
		assert(results[0].shape[0] == 20)

	def test_save_model(self):
		# should save and re-load model
		tmp_path = self.tmpFile()
		self.model.save(tmp_path)
		loaded_model = VLP(checkpoint=tmp_path, data = self.model.data)
		results = loaded_model.test(self.validation_size)
		assert(results['preds_mean'].shape[0] == self.validation_size)
