import os
import sys
import tempfile
import unittest
from epitome.models import *

# set Epitome data path to test data files for testing
# this data was saved using functions.saveToyData(<epitome_repo_path>/epitome/test/daata)
dir_path = os.path.dirname(os.path.realpath(__file__))
os.environ["EPITOME_DATA_PATH"] = os.path.abspath(os.path.join(dir_path, "data"))

S3_TEST_PATH = 'https://epitome-data.s3-us-west-1.amazonaws.com/test/data.zip'

# set Epitome data path to test data files for testing
dir_path = os.path.dirname(os.path.realpath(__file__))
os.environ["EPITOME_DATA_PATH"] = os.path.abspath(os.path.join(dir_path, "data"))

class EpitomeTestCase(unittest.TestCase):

	def __init__(self, *args, **kwargs):
		# download test data to parent dir of EPITOME_DATA_PATH  if it was not yet downloaded
		download_and_unzip(S3_TEST_PATH, os.path.dirname(os.environ["EPITOME_DATA_PATH"]))
		super(EpitomeTestCase, self).__init__(*args, **kwargs)

	def getFeatureData(self, eligible_assays, eligible_cells, similarity_assays = ['DNase']):
		# returns matrix, cellmap, assaymap
		return get_assays_from_feature_file(
				eligible_assays = eligible_assays,
				similarity_assays = similarity_assays,
				eligible_cells = eligible_cells, min_cells_per_assay = 3, min_assays_per_cell = 1)

	def makeSmallModel(self):

		eligible_cells = ['K562','HepG2','H1','A549','HeLa-S3']
		eligible_assays = ['DNase','CTCF']
		matrix, cellmap, assaymap = self.getFeatureData(eligible_assays, eligible_cells)

		return VLP(list(eligible_assays),
			test_celltypes = ['K562'],
			matrix = matrix,
			assaymap = assaymap,
			cellmap = cellmap)


	def tmpFile(self):

		tempFile = tempfile.NamedTemporaryFile(delete=True)
		tempFile.close()
		return tempFile.name
