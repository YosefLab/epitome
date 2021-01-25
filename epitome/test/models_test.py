from epitome.test import EpitomeTestCase
from epitome.constants import Dataset
import numpy as np
from epitome.models import VLP
import pytest
import tempfile
import pyranges as pr

class ModelsTest(EpitomeTestCase):

	def __init__(self, *args, **kwargs):
		super(ModelsTest, self).__init__(*args, **kwargs)
		self.model = self.makeSmallModel()
		self.validation_size = 10

	def test_score_peak_file(self):
		test_similarity_peak_file = tempfile.NamedTemporaryFile(delete=False)
		test_regions_peak_file = tempfile.NamedTemporaryFile(delete=False)

		# Create dummy data
		similarity_dict =  {'Chromosome': ['chr1', 'chr1', 'chr6'], 'Start': [200, 400, 1100],  'End': [220, 440, 1150]}
		regions_dict = {'Chromosome': ['chr1', 'chr1'], 'Start': [210, 410],  'End': [215, 415]}
		similarity_pr = pr.from_dict(similarity_dict)
		regions_pr = pr.from_dict(regions_dict)

		# Write to temp bed file
		similarity_pr.to_bed(test_similarity_peak_file.name)
		regions_pr.to_bed(test_regions_peak_file.name)

		test_similarity_peak_file.flush()
		test_regions_peak_file.flush()

		preds = self.model.score_peak_file([test_similarity_peak_file.name],
			test_regions_peak_file.name,
			all_data=None)

		test_regions_peak_file.close()
		test_similarity_peak_file.close()

		assert(preds.shape[0] == len(regions_pr))

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

	def test_test_model(self):

		# make sure can run in test mode
		results = self.model.test(self.validation_size, mode=Dataset.TEST)
		assert(results['preds_mean'].shape[0] == self.validation_size)

	def test_specify_assays(self):
		# test for https://github.com/YosefLab/epitome/issues/23
		# should add DNase to eligible assays

		eligible_assays = ['CTCF', 'RAD21', 'CEBPB']

		model = VLP(list(eligible_assays))
		assert(len(model.assaymap) == 4)

	def test_model_similarity_assays(self):
		# should train a model without using DNAse
		eligible_assays = ['CTCF', 'RAD21', 'CEBPB']

		model = VLP(list(eligible_assays), similarity_assays = ['H3K27ac'])
		assert(len(model.assaymap) == 4)

	def test_model_two_similarity_assays(self):
		# should train a model without using DNAse
		eligible_assays = ['CTCF', 'RAD21', 'CEBPB']

		model = VLP(list(eligible_assays), similarity_assays = ['DNase', 'H3K27ac'])
		assert(len(model.assaymap) == 5)

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

	def test_score_matrix(self):

		regions_peak_file = tempfile.NamedTemporaryFile(delete=False)

		# Create dummy data
		regions_dict = {'Chromosome': ['chr1', 'chr1'],
						'Start': [10000, 30000],
						'End': [10300, 31200]}
		regions_pr = pr.from_dict(regions_dict)

		# Write to tmp bed file
		regions_pr.to_bed(regions_peak_file.name)
		regions_peak_file.flush()

		accessilibility_peak_matrix = np.random.uniform(low=0., high=1., size=(4,2))

		results = self.model.score_matrix(accessilibility_peak_matrix,
								regions_peak_file.name, all_data = None)

		assert(results.shape == (4, 2, 1))

	def test_score_matrix_missing_data(self):
		# if there is a region in the regions file that does not overlap anything
		# in the training data, it should return ??

		regions_peak_file = tempfile.NamedTemporaryFile(delete=False)

		# Create dummy data
		regions_dict = {'Chromosome': ['chr1', 'chr1'],
						'Start': [50, 10000],
						'End': [150, 10400]}

		regions_pr = pr.from_dict(regions_dict)

		# Write to tmp bed file
		regions_pr.to_bed(regions_peak_file.name)
		regions_peak_file.flush()

		accessilibility_peak_matrix = np.random.uniform(low=0., high=1., size=(4,2))

		results = self.model.score_matrix(accessilibility_peak_matrix,
											regions_peak_file.name,
											all_data = None)

		assert np.all(np.isnan(results[:,0,:]))
