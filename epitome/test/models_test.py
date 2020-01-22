from epitome.test import EpitomeTestCase
from epitome.constants import Dataset
import numpy as np
from epitome.models import VLP

class ModelsTest(EpitomeTestCase):

	def test_model_functions(self):
		train_iters = 50
		validation_size = 10
		
		# create model and train
		model = self.makeSmallModel()
		model.train(train_iters)
		results = model.test(validation_size)

		# Make sure predictions are not random
		# after first iterations
		assert(results['preds_mean'].numpy()[0,0] < 0.1)
		assert(results['preds_mean'].shape[0] == validation_size)


		# make sure can run in test mode
		results = model.test(validation_size, mode=Dataset.TEST)
		assert(results['preds_mean'].numpy()[0,0] < 0.1)

		# should be able to evaluate on a dnase vector
		dnase_vector = np.ones(model.data[Dataset.TRAIN].shape[1])
		results = model.eval_vector(model.data[Dataset.TRAIN], dnase_vector, np.arange(0,20))
		assert(results[0].shape[0] == 20)

		# should save and re-load model
		tmp_path = self.tmpFile()
		model.save(tmp_path)
		loaded_model = VLP(data_path = model.data_path, checkpoint=tmp_path)
		results = loaded_model.test(validation_size)
		assert(results['preds_mean'].numpy()[0,0] < 0.1)
