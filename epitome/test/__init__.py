import os
import sys
import tempfile
import unittest
from epitome.models import *
from epitome import GET_DATA_PATH

class EpitomeTestCase(unittest.TestCase):

	def makeSmallModel(self):

		data_path = GET_DATA_PATH()
		x = os.path.join(data_path, 'valid.npz')
		# load in small validation matrix for test
		sparse_matrix = scipy.sparse.load_npz(x).toarray()
		data = {Dataset.TRAIN: sparse_matrix, Dataset.VALID: sparse_matrix, Dataset.TEST: sparse_matrix}

		
		eligible_cells = ['K562','HepG2','H1','A549','HeLa-S3']
		eligible_assays = ['DNase','CTCF']
		matrix, cellmap, assaymap = get_assays_from_feature_file(os.path.join(data_path, "feature_name"),
                                                         eligible_assays = eligible_assays,
                                  eligible_cells = eligible_cells, min_cells_per_assay = 3, min_assays_per_cell = 1)

		return VLP(data_path,
            		['K562'],
            		matrix,
            		assaymap,
            		cellmap,
            		shuffle_size=2, 
            		batch_size=64, data = data)


	def tmpFile(self):

		tempFile = tempfile.NamedTemporaryFile(delete=True)
		tempFile.close()
		return tempFile.name
