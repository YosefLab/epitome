from epitome.test import EpitomeTestCase
import numpy as np
from epitome.generators import load_data
from epitome.constants import Dataset
from epitome.dataset import EpitomeDataset

class GeneratorsTest(EpitomeTestCase):

	def __init__(self, *args, **kwargs):
		super(GeneratorsTest, self).__init__(*args, **kwargs)

		# get shape of train data
		eligible_cells = ['K562','HepG2','H1','A549','HeLa-S3']
		eligible_targets = ['DNase','CTCF']
		matrix, cellmap, targetmap = self.getFeatureData(eligible_targets, eligible_cells)
		self.train_shape = (np.max(matrix)+1, 1800)  # 1800 is size of toy data training

	def test_generator_no_dnase(self):

        # generate consistent data
		data = np.zeros(self.train_shape)
		data[::2] = 1 # every 2nd row is 1s

		eligible_cells = ['K562','HepG2','H1','A549','HeLa-S3']
		eligible_targets = ['DNase','CTCF']
		matrix, cellmap, targetmap = self.getFeatureData(eligible_targets, eligible_cells)


		label_cell_types = ['K562']
		eligible_cells.remove(label_cell_types[0])


		results = load_data(data,
			['K562'],
			eligible_cells,
			matrix,
			targetmap,
			cellmap,
			radii = [], # no dnase
			mode = Dataset.VALID,
			indices=np.arange(0,10))()
		li_results = list(results)

		# this element is a positive
		pos_position = 6
		print(li_results[pos_position][-2])
		assert(np.all(li_results[pos_position][-2] == 1))


	def test_generator_no_similarity(self):
		# generate consistent data
		data = np.zeros(self.train_shape)
		data[::2] = 1 # every 2nd row is 1s

		eligible_cells = ['K562','HepG2','H1','HeLa-S3']
		eligible_targets = ['CTCF']
		matrix, cellmap, targetmap = self.getFeatureData(eligible_targets, eligible_cells, similarity_targets = [])
		assert(len(list(targetmap)) == 1) # should not have added DNase

		label_cell_types = ['K562']
		eligible_cells.remove(label_cell_types[0])

		results = load_data(data,
			['K562'],
			eligible_cells,
			matrix,
			targetmap,
			cellmap,
			radii = [],
			mode = Dataset.VALID,
			similarity_targets = [],
			indices=np.arange(0,2), return_feature_names = True)()

		li_results = list(results)
		labels = li_results[0][1][0]
		assert(labels[0] =='HepG2_CTCF')

	def test_generator_only_H3(self):

		# generate consistent data
		data = np.zeros(self.train_shape)
		data[::2] = 1 # every 2nd row is 1s

		eligible_cells = ['K562','HepG2','H1','HeLa-S3']
		eligible_targets = ['H3K27ac','CTCF']
		matrix, cellmap, targetmap = self.getFeatureData(eligible_targets, eligible_cells, similarity_targets = ['H3K27ac'])
		assert(len(list(targetmap)) == 2) # should not have added DNase

		label_cell_types = ['K562']
		eligible_cells.remove(label_cell_types[0])

		results = load_data(data,
			['K562'],
			eligible_cells,
			matrix,
			targetmap,
			cellmap,
			radii = [1,3],
			mode = Dataset.VALID,
			similarity_targets = ['H3K27ac'],
			indices=np.arange(0,2), return_feature_names = True)()

		li_results = list(results)
		labels = li_results[0][1][0]
		assert(labels[0] =='HepG2_H3K27ac')

	def test_generator_sparse_data(self):

		eligible_cells = ['K562','HepG2','H1','A549','HeLa-S3']
		eligible_targets = ['DNase','CTCF','RAD21','LARP7']
		dataset = EpitomeDataset(targets = eligible_targets,
			cells = eligible_cells,
			min_cells_per_target = 1,
			min_targets_per_cell = 1,
			data_dir=self.epitome_data_dir,
			assembly=self.epitome_assembly)


		label_cell_types = ['HepG2']
		eligible_cells.remove(label_cell_types[0])

		results = list(load_data(dataset.get_data(Dataset.TRAIN),
			label_cell_types,
			eligible_cells,
			dataset.matrix,
			dataset.targetmap,
			dataset.cellmap,
			radii = [],
			mode = Dataset.VALID,
			return_feature_names=True,
			indices=np.arange(0,10))())

		# get first features
		features = results[0][0]

		# get labels
		labels = results[0][1]

		#  all cell types but K562 are missing LARP7 data
		assert(len(features[0]) == len(eligible_cells) * len(eligible_targets) - 3)

		# make sure mask is masking out LARP7 for HepG2
		assert(np.all(features[-1] == [1., 0., 1.]))

		# make sure first label cell is not the test cell K562
		assert(labels[-2][0] == 'lbl_HepG2_RAD21')
		assert(labels[-2][1] == 'lbl_HepG2_LARP7')
		assert(labels[-2][2] == 'lbl_HepG2_CTCF')

	def test_generator_radius(self):
		eligible_cells = ['K562','HepG2','H1','A549','HeLa-S3']
		eligible_targets = ['DNase','CTCF','RAD21']

		dataset = EpitomeDataset(targets = eligible_targets,
			cells = eligible_cells,
			data_dir=self.epitome_data_dir,
			assembly=self.epitome_assembly)

		label_cell_types = ['K562']
		eligible_cells.remove(label_cell_types[0])

		radii = [1,10]

		results = load_data(dataset.get_data(Dataset.TRAIN),
			['K562'],
			eligible_cells,
			dataset.matrix,
			dataset.targetmap,
			dataset.cellmap,
			radii = radii,
			mode = Dataset.VALID,
			indices=np.arange(0,10))()
		li_results = list(results)

		# length should include eligible targets and 2* radius for pos and agreement
		assert(len(li_results[0][0]) == len(eligible_cells) * (len(eligible_targets)+len(radii)* 2))

	def test_generator_multiple_sim(self):
		eligible_cells = ['K562','HepG2','H1','A549','HeLa-S3']
		eligible_targets = ['DNase','CTCF','RAD21']
		dataset = EpitomeDataset(targets = eligible_targets,
					cells = eligible_cells,
					data_dir=self.epitome_data_dir,
					assembly=self.epitome_assembly)

		label_cell_types = ['K562']
		eligible_cells.remove(label_cell_types[0])

		similarity_matrix = np.ones([2,dataset.get_data(Dataset.TRAIN).shape[1]])

		radii = [1,10]

		results = load_data(dataset.get_data(Dataset.TRAIN),
			['K562'],
			eligible_cells,
			dataset.matrix,
			dataset.targetmap,
			dataset.cellmap,
			radii = radii,
			mode = Dataset.RUNTIME,
			similarity_matrix = similarity_matrix,
			similarity_targets = ['DNase','CTCF'],
			indices=np.arange(0,10))()
		li_results = list(results)

		# length should include eligible targets and 2* radius for pos and agreement
		# for each of the 2 similarity targets
		assert(len(li_results[0][0]) == len(eligible_cells) * (len(eligible_targets)+len(radii)* 4))

	def test_generator_dnase_array(self):
		# should not fail if similarity_targets are just for DNase and is a single array.
		# https://github.com/YosefLab/epitome/issues/4
		eligible_cells = ['K562','HepG2','H1','A549','HeLa-S3']
		eligible_targets = ['DNase','CTCF','RAD21','LARP7']
		dataset = EpitomeDataset(targets = eligible_targets,
							cells = eligible_cells,
							data_dir=self.epitome_data_dir,
							assembly=self.epitome_assembly)
		test_celltypes = ['K562']
		eligible_cells.remove(test_celltypes[0])
		radii = [1,10]
		# fake data for DNase
		similarity_matrix = np.ones(dataset.get_data(Dataset.TRAIN).shape[1])

		results = load_data(dataset.get_data(Dataset.TRAIN),
	             test_celltypes,
	             eligible_cells,
	             dataset.matrix,
	             dataset.targetmap,
	             dataset.cellmap,
				 radii,
	             mode = Dataset.RUNTIME,
	             similarity_matrix = similarity_matrix,
	             similarity_targets = 'DNase',
				 return_feature_names = True,
	             indices=np.arange(0,10))()
		li_results = list(results)

		# if we reach here, an error was not thrown :)
		assert(len(li_results) == 10)
		assert(len(li_results[0][0][0]) == 28)

		feature_names = li_results[0][1][0]
		assert(len(list(filter(lambda x: '_agree' in x, feature_names))) == len(radii) * len(eligible_cells))


	def test_continous_generator(self):
		eligible_cells = ['K562','HepG2','H1','A549','HeLa-S3']
		eligible_targets = ['DNase','CTCF','RAD21','LARP7']
		dataset = EpitomeDataset(targets = eligible_targets,
							cells = eligible_cells,
							data_dir=self.epitome_data_dir,
							assembly=self.epitome_assembly)
		test_celltypes = ['K562']
		eligible_cells.remove(test_celltypes[0])
		radii = [1,10]
		# fake data for DNase
		similarity_matrix = np.ones(dataset.get_data(Dataset.TRAIN).shape[1])

		results = load_data(dataset.get_data(Dataset.TRAIN),
	             test_celltypes,
	             eligible_cells,
	             dataset.matrix,
	             dataset.targetmap,
	             dataset.cellmap,
				 radii,
	             mode = Dataset.RUNTIME,
				 continuous = True,
	             similarity_matrix = similarity_matrix,
	             similarity_targets = 'DNase',
				 return_feature_names = True,
	             indices=np.arange(0,10))()
		li_results = list(results)
		feature_names = li_results[0][1][0]

		# make sure we don't have the agreement features
		assert(len(li_results[0][0][0]) == 20)
		assert(len(list(filter(lambda x: '_agree' in x, feature_names))) == 0)
