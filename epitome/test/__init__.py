import os
import sys
import tempfile
import unittest
from epitome.models import *

S3_TEST_PATH = 'https://epitome-data.s3-us-west-1.amazonaws.com/test/data.zip'

# set Epitome data path to test data files for testing
dir_path = os.path.dirname(os.path.realpath(__file__))
os.environ["EPITOME_DATA_PATH"] = os.path.abspath(os.path.join(dir_path, "data"))

class EpitomeTestCase(unittest.TestCase):

	def getValidDataShape(self):
		# number of rows in feature_name file by n random genome regions
		return (749, 50000)

	def getValidData(self):
		np.random.seed(1)
		# generate striped array
		return np.random.randint(2, size=self.getValidDataShape())

	def getFeatureData(self, eligible_assays, eligible_cells):
		# returns matrix, cellmap, assaymap
		return get_assays_from_feature_file(feature_name_file = 'epitome/test/data/feature_name',
				eligible_assays = eligible_assays,
				eligible_cells = eligible_cells, min_cells_per_assay = 3, min_assays_per_cell = 1)

	def makeSmallModel(self):

		sparse_matrix = self.getValidData()
		data = {Dataset.TRAIN: sparse_matrix, Dataset.VALID: sparse_matrix, Dataset.TEST: sparse_matrix}


		eligible_cells = ['K562','HepG2','H1','A549','HeLa-S3']
		eligible_assays = ['DNase','CTCF']
		matrix, cellmap, assaymap = self.getFeatureData(eligible_assays, eligible_cells)

		return VLP(list(eligible_assays),
			test_celltypes = ['K562'],
			matrix = matrix,
			assaymap = assaymap,
			cellmap = cellmap,
			data = data)


	def tmpFile(self):

		tempFile = tempfile.NamedTemporaryFile(delete=True)
		tempFile.close()
		return tempFile.name
