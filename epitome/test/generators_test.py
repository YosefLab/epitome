from epitome.test import EpitomeTestCase
import numpy as np
from epitome.generators import *

class GeneratorsTest(EpitomeTestCase):

	def test_generator_no_dnase(self):

		data = self.getValidData()

		eligible_cells = ['K562','HepG2','H1','A549','HeLa-S3']
		eligible_assays = ['DNase','CTCF']
		matrix, cellmap, assaymap = self.getFeatureData(eligible_assays, eligible_cells)

		label_cell_types = ['K562']
		eligible_cells.remove(label_cell_types[0])


		results = load_data(data,
			['K562'],
			eligible_cells,
			matrix,
			assaymap,
			cellmap,
			radii = [],
			mode = Dataset.VALID,
			indices=np.arange(0,10))()
		li_results = list(results)

		# this element is a positive
		pos_position = 6
		score = data[matrix[cellmap['K562'],assaymap['CTCF']]][pos_position]
		assert(np.all(li_results[pos_position][-2] == 1))

	def test_generator_sparse_data(self):
		data = self.getValidData()
		eligible_cells = ['K562','HepG2','H1','A549','HeLa-S3']
		eligible_assays = ['DNase','CTCF','RAD21','LARP7']
		matrix, cellmap, assaymap = self.getFeatureData(eligible_assays, eligible_cells)
		label_cell_types = ['K562']
		eligible_cells.remove(label_cell_types[0])

		results = load_data(data,
			['K562'],
			eligible_cells,
			matrix,
			assaymap,
			cellmap,
			radii = [],
			mode = Dataset.VALID,
			indices=np.arange(0,10))()
		li_results = list(results)

		# length should be shorter for first cell because missing LARP7
		assert(len(li_results[0][0]) == len(eligible_assays)-1)

	def test_generator_radius(self):
		data = self.getValidData()
		eligible_cells = ['K562','HepG2','H1','A549','HeLa-S3']
		eligible_assays = ['DNase','CTCF','RAD21']
		matrix, cellmap, assaymap = self.getFeatureData(eligible_assays, eligible_cells)
		label_cell_types = ['K562']
		eligible_cells.remove(label_cell_types[0])

		radii = [1,10]

		results = load_data(data,
			['K562'],
			eligible_cells,
			matrix,
			assaymap,
			cellmap,
			radii = radii,
			mode = Dataset.VALID,
			indices=np.arange(0,10))()
		li_results = list(results)

		# length should include eligible assays and 2* radius for pos and agreement
		assert(len(li_results[0][0]) == len(eligible_assays)+len(radii)* 2)


	def test_generator_runtime(self):
		data = self.getValidData()
		eligible_cells = ['K562','HepG2','H1','A549','HeLa-S3']
		eligible_assays = ['DNase','CTCF','RAD21']
		matrix, cellmap, assaymap = self.getFeatureData(eligible_assays, eligible_cells)
		label_cell_types = ['K562']
		eligible_cells.remove(label_cell_types[0])

		similarity_matrix = np.ones([2,data.shape[1]])

		radii = [1,10]

		results = load_data(data,
			['K562'],
			eligible_cells,
			matrix,
			assaymap,
			cellmap,
			radii = radii,
			mode = Dataset.RUNTIME,
			similarity_matrix = similarity_matrix,
			similarity_assays = ['DNase','CTCF'],
			indices=np.arange(0,10))()
		li_results = list(results)

		# length should include eligible assays and 2* radius for pos and agreement
		assert(len(li_results[0][0]) == len(eligible_assays)+len(radii)* 4)
