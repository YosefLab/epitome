import os
import sys
import tempfile
import unittest
from epitome.models import *
from epitome.dataset import *

S3_TEST_PATH = 'https://epitome-data.s3-us-west-1.amazonaws.com/test.zip'
dir_path = os.path.dirname(os.path.realpath(__file__))
TEST_DATA_PATH = os.path.abspath(os.path.join(dir_path, "data","test"))

class EpitomeTestCase(unittest.TestCase):

	def __init__(self, *args, **kwargs):
		# EpitomeTestCase.setEpitomeDataPath()
		# download test data to parent dir of EPITOME_DATA_PATH  if it was not yet downloaded
		# download_and_unzip(S3_TEST_PATH, os.path.dirname(os.environ["EPITOME_DATA_PATH"]))
		download_and_unzip(S3_TEST_PATH, TEST_DATA_PATH)
		super(EpitomeTestCase, self).__init__(*args, **kwargs)

	@staticmethod
	def getEpitomeTestDataPath():
		return TEST_DATA_PATH

	@staticmethod
	def setEpitomeDataPath():

		# set Epitome data path to test data files for testing
		# this data was saved using functions.saveToyData(<epitome_repo_path>/epitome/test/data)
		dir_path = os.path.dirname(os.path.realpath(__file__))
		# os.environ["EPITOME_DATA_PATH"] = os.path.abspath(os.path.join(dir_path, "data","test"))
		return os.path.abspath(os.path.join(dir_path, "data","test"))

	def getFeatureData(self,
					targets,
					cells,
					similarity_targets = ['DNase'],
					min_cells_per_target = 3,
					min_targets_per_cell = 1):

		# returns matrix, cellmap, assaymap
		return EpitomeDataset.get_assays(
				targets = targets,
				cells = cells,
				data_dir = TEST_DATA_PATH,
				similarity_targets = similarity_targets,
				min_cells_per_target = min_cells_per_target,
				min_targets_per_cell = min_targets_per_cell)

	def makeSmallModel(self):
		eligible_cells = ['K562','HepG2','H1','A549','HeLa-S3']
		eligible_targets = ['DNase','CTCF']

		dataset = EpitomeDataset(targets = eligible_targets,
			cells = eligible_cells, data_dir=TEST_DATA_PATH)


		return EpitomeModel(dataset,
			test_celltypes = ['K562'])

	def tmpFile(self):
		tempFile = tempfile.NamedTemporaryFile(delete=True)
		tempFile.close()
		return tempFile.name
