import os
import sys
import tempfile
import unittest
from epitome.models import *
from epitome.dataset import *

class EpitomeTestCase(unittest.TestCase):

	def __init__(self, *args, **kwargs):
		# class definitions
		dir_path = os.path.dirname(os.path.realpath(__file__))
		self.epitome_assembly = "test"
		self.epitome_data_dir = os.path.abspath(os.path.join(dir_path, "data"))
		self.epitome_test_dir = os.path.abspath(os.path.join(self.epitome_data_dir, self.epitome_assembly))
		super(EpitomeTestCase, self).__init__(*args, **kwargs)

	def getFeatureData(self,
					targets,
					cells,
					similarity_targets = ['DNase'],
					min_cells_per_target = 3,
					min_targets_per_cell = 1):

		return EpitomeDataset.get_assays(
				targets = targets,
				cells = cells,
				data_dir = self.epitome_data_dir,
				assembly = self.epitome_assembly,
				similarity_targets = similarity_targets,
				min_cells_per_target = min_cells_per_target,
				min_targets_per_cell = min_targets_per_cell)

	def makeSmallModel(self):
		eligible_cells = ['K562','HepG2','H1','A549','HeLa-S3']
		eligible_targets = ['DNase','CTCF']

		dataset = EpitomeDataset(targets = eligible_targets,
								 cells = eligible_cells,
								 data_dir=self.epitome_data_dir,
								 assembly=self.epitome_assembly)

		return EpitomeModel(dataset,
			test_celltypes = ['K562'])

	def tmpFile(self):
		tempFile = tempfile.NamedTemporaryFile(delete=True)
		tempFile.close()
		return tempFile.name
