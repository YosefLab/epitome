from epitome.test import EpitomeTestCase
from epitome.constants import Dataset

class ModelsTest(EpitomeTestCase):

	def test_trainModel(self):
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
