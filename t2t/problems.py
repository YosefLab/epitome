import math
import multiprocessing as mp
import os
import fnmatch
import sys

# import local accessibility
sys.path.insert(0, os.path.abspath('./data'))
from accessibility import get_accessibility_vector
from accessibility import get_accessibility_vector_pybed
from accessibility import save_merged_bedfile


import tarfile
import numpy as np
import pandas as pd
import linecache

# Dependency imports

import h5py
import pybedtools
import numpy as np
from scipy.io import loadmat
import fnmatch

from tensor2tensor.data_generators import dna_encoder
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.utils import metrics
from tensor2tensor.utils import registry

import tensorflow as tf


@registry.register_problem
class ProteinBindingProblem(problem.Problem):
	"""Transcription factor binding site prediction."""

	@property
	def num_output_predictions(self):
		"""Number of binding site predictions."""
		return 919

	@property
	def sequence_length(self):
		"""Sequence length."""
		return 1000

	@property
	def num_channels(self):
		"""Input channels."""
		return 4

	@property
	def num_classes(self):
		"""Binary classification."""
		return 1

	@property
	def num_shards(self):
		"""Number of TF-Record shards."""
		return 100

	def generator(self, tmp_dir, is_training):
		raise NotImplementedError()

	def generate_data(self, data_dir, tmp_dir, task_id=-1):
		generator_utils.generate_dataset_and_shuffle(
			self.generator(tmp_dir, is_training=True),
			self.training_filepaths(data_dir, self.num_shards, shuffled=True),
			self.generator(tmp_dir, is_training=False),
			self.dev_filepaths(data_dir, 1, shuffled=True))

	def hparams(self, defaults, unused_model_hparams):
		p = defaults
		# TODO(alexyku): is it okay to use None here?
		p.input_modality = {"inputs": (registry.Modalities.GENERIC, None)}
		p.target_modality = (registry.Modalities.SYMBOL, self.num_classes)
		p.input_space_id = problem.SpaceID.GENERIC
		p.target_space_id = problem.SpaceID.GENERIC
        
        
@registry.register_problem
class DeepSeaProblem(ProteinBindingProblem):
	"""Transcription factor binding site prediction."""

	_DEEPSEA_DOWNLOAD_URL = ("http://deepsea.princeton.edu/media/code/"
							 "deepsea_train_bundle.v0.9.tar.gz")
	_DEEPSEA_FILENAME = "deepsea_train_bundle.v0.9.tar.gz"
	_DEEPSEA_DIRNAME = "deepsea_train"
	_DEEPSEA_TRAIN_FILENAME = "deepsea_train/train.mat"
	_DEEPSEA_TEST_FILENAME = "deepsea_train/valid.mat"

	def generator(self, tmp_dir, is_training):
		self._get_data(tmp_dir)
		if is_training:
			return self._train_generator(tmp_dir)
		else:
			return self._test_generator(tmp_dir)

	def _get_data(self, directory):
		"""Download all Deepsea files to directory unless they are there."""
		generator_utils.maybe_download(
			directory, self._DEEPSEA_FILENAME, self._DEEPSEA_DOWNLOAD_URL)
		tar_name = os.path.join(directory, self._DEEPSEA_FILENAME)
		tar = tarfile.open(tar_name, "r:gz")
		tar.extractall(directory)
		tar.close()

	def _train_generator(self, tmp_dir):
		tmp = h5py.File(os.path.join(tmp_dir, self._DEEPSEA_TRAIN_FILENAME))
		all_inputs, all_targets = tmp['trainxdata'], tmp['traindata']

		for i in range(all_inputs.shape[2]):
			inputs = all_inputs[:, :, i]
			targets = np.expand_dims(all_targets[:, i], -1)

			yield {
				"inputs": [inputs.astype(np.bool).tobytes()],
				"targets": [targets.astype(np.bool).tobytes()]
			}

	def _test_generator(self, tmp_dir):
		tmp = loadmat(os.path.join(tmp_dir, self._DEEPSEA_TEST_FILENAME))
		all_inputs, all_targets = tmp['validxdata'], tmp['validdata']

		for i in range(all_inputs.shape[0]):
			inputs = all_inputs[i].transpose([1, 0])
			targets = np.expand_dims(all_targets[i], -1)

			yield {
				"inputs": [inputs.astype(np.bool).tobytes()],
				"targets": [targets.astype(np.bool).tobytes()]
			}

	def example_reading_spec(self):
		data_fields = {
			"inputs": tf.FixedLenFeature([], tf.string),
			"targets": tf.FixedLenFeature([], tf.string)
		}
		data_items_to_decoders = None
		return (data_fields, data_items_to_decoders)

	def preprocess_example(self, example, mode, unused_hparams):
		del mode

		inputs = example["inputs"]
		targets = example["targets"]

		inputs_shape = [self.sequence_length, self.num_channels]
		targets_shape = [self.num_output_predictions, self.num_classes]

		# Parse the bytestring based on how you encoded it in common_generator
		inputs = tf.reshape(tf.decode_raw(inputs, tf.bool), inputs_shape)
		targets = tf.reshape(tf.decode_raw(targets, tf.bool), targets_shape)

		example["inputs"] = tf.to_float(inputs)
		example["targets"] = tf.to_int32(targets)
		return example


@registry.register_problem
class EpitomeProblem(ProteinBindingProblem):
	_DEEPSEA_DOWNLOAD_URL = ("http://deepsea.princeton.edu/media/code/"
							 "deepsea_train_bundle.v0.9.tar.gz")
	_DEEPSEA_FILENAME = "deepsea_train_bundle.v0.9.tar.gz"
	_DEEPSEA_DIRNAME = "deepsea_train"
	_DEEPSEA_TRAIN_FILENAME = "deepsea_train/train.mat"
	_DEEPSEA_TEST_FILENAME = "deepsea_train/valid.mat"
	_DEEPSEA_FEATURES_FILENAME = "../data/feature_name"
	_DNASE_BED_DIRNAME = "/data/epitome/accessibility/dnase/hg19"
	_JOINED_PY_BED_EXTENSION = "_joined_bedtools.bed"

	_DEEPSEA_GENOME_REGIONS_URL = ("http://deepsea.princeton.edu/media/code/"
							 "allTFs.pos.bed.tar.gz")
	_DEEPSEA_GENOME_REGIONS_TAR_FILENAME = "allTFs.pos.bed.tar.gz"
	_DEEPSEA_GENOME_REGIONS_FILENAME = "allTFs.pos.bed"
	_TRAIN_REGIONS = [0, 2200000-1]
	_VALID_REGIONS = [2200001-1, 2204000-1]
	_TEST_REGIONS  = [2204001-1, 2608182-1]

    
	@property
	def train_cells(self):
		# available cell types for hg19 on c66: ['H1-hESC', 'HeLa-S3', 'GM12878',  'HepG2',  'K562', 'A549']
		return ['H1-hESC', 'HeLa-S3', 'GM12878',  'HepG2'] 

	@property
	def test_cells(self):
		return ['K562']

	@property
	def train_proteins(self):
		return ['p300', 'NRSF', 'CTCF', 'GABP', 'JunD', 'CEBPB', 'Pol2', 'EZH2',
				'Mxi1', 'Max', 'RFX5', 'TAF1', 'Nrf1', 'Rad21', 'TBP', 'USF2',
				'c-Myc','CHD2']
    
	@property
	def inputs_shape(self):
		return [1000, 6] # shape of each record

	@property
	def num_examples(self):
		# there will be a separate example for each region of the genome for each cell type
		return 4400000

	@property
	def num_output_predictions(self):
		"""Number of binding site predictions."""
		return len(self.train_proteins)

	@property
	def num_channels(self):
		"""Input channels."""
		""" Channel for sequence data (4), accessibility data (1) and strand. """
		return 6

	def _get_data(self, directory):
		"""Download all files to directory unless they are there."""
		generator_utils.maybe_download(
			directory, self._DEEPSEA_FILENAME, self._DEEPSEA_DOWNLOAD_URL)
		tar_name = os.path.join(directory, self._DEEPSEA_FILENAME)
		tar = tarfile.open(tar_name, "r:gz")
		tar.extractall(directory)
		tar.close()
        
		generator_utils.maybe_download(
			directory, self._DEEPSEA_GENOME_REGIONS_TAR_FILENAME, self._DEEPSEA_GENOME_REGIONS_URL)
		tar_name = os.path.join(directory, self._DEEPSEA_GENOME_REGIONS_TAR_FILENAME)
		tar = tarfile.open(tar_name, "r:gz")
		tar.extractall(directory)
		tar.close()

	def generator(self, tmp_dir, is_training):
		self._get_data(tmp_dir)
		# TODO: training, validation and test
		if is_training:
			return self._train_generator(tmp_dir)
		else:
			return self._validation_generator(tmp_dir)
                    
                    
	def _train_generator(self, tmp_dir):
		tmp = h5py.File(os.path.join(tmp_dir, self._DEEPSEA_TRAIN_FILENAME))
		all_inputs, all_targets = tmp['trainxdata'], tmp['traindata']
        
		for cell in self.train_cells:
			tf.logging.info("Generating training data for cell %s" % (cell))
			for example in self.cell_generator(tmp_dir, all_inputs, all_targets, cell, self._TRAIN_REGIONS):
				yield example


	def _validation_generator(self, tmp_dir):
		tmp = loadmat(os.path.join(tmp_dir, self._DEEPSEA_TEST_FILENAME))
		all_inputs, all_targets = tmp['validxdata'], tmp['validdata']
		inputs = all_inputs.transpose(2, 1, 0)
		targets = all_targets.transpose()
		for cell in self.test_cells:
			tf.logging.info("Generating testing data for cell %s" % (cell))	
			for example in self.cell_generator(tmp_dir, inputs, targets, cell, self._VALID_REGIONS):
				yield example

	def _test_generator(self):
		# TODO
		raise NotImplementedError()		
                    
	def example_reading_spec(self):
		data_fields = {
			'chr': tf.FixedLenFeature([], tf.string),
			'start': tf.FixedLenFeature([], tf.string),
			'stop': tf.FixedLenFeature([], tf.string),
			"inputs/data": tf.FixedLenFeature([], tf.string),
			"targets": tf.FixedLenFeature([], tf.string),
			"mask": tf.FixedLenFeature([], tf.string)
		}
		data_items_to_decoders = None
		return (data_fields, data_items_to_decoders)

	def preprocess_example(self, example, mode, unused_hparams):
		del mode

		chr_ = example["chr"]
		start = example["start"]
		stop = example["stop"]
		inputs = example["inputs/data"]
		targets = example["targets"]
		mask = example["mask"]

		targets_shape = [self.num_output_predictions, 1]

		# Parse the bytestring based on how you encoded it in common_generator
		inputs = tf.reshape(tf.decode_raw(inputs, tf.int32),
							tf.decode_raw(self.inputs_shape, tf.int32))
		targets = tf.reshape(tf.decode_raw(targets, tf.bool), targets_shape)
		mask = tf.reshape(tf.decode_raw(mask, tf.bool), targets_shape)
        

		example["chr"] = tf.decode_raw(chr_, tf.uint8)[0]
		example["start"] = tf.decode_raw(start, tf.int64)[0]
		example["stop"] = tf.decode_raw(stop, tf.int64)[0]
		example["inputs"] = tf.to_float(inputs)
		example["targets"] = tf.to_int32(targets)
		example["mask"] = tf.to_int32(mask)

		return example


	def cell_generator(self, tmp_dir, all_inputs, all_targets, cell, indices):
		''' Builds dicts of indicies of different features
		:param all_inputs inputs
		:param all_targets targets
		:param cell target cell
		:param indices indices of line number in all positions file (_DEEPSEA_GENOME_REGIONS_FILENAME )
		'''
		start = indices[0]
		stop  = indices[1]
        
        all_targets
    
		dnase_dict, tf_dict = self.parse_feature_name(self._DEEPSEA_FEATURES_FILENAME)
        
		# builds the vector of locations for querying from matrix, and the mask
		tf_locs, tf_mask = self.get_feature_indices(tf_dict, cell)

		# Pre-build these features
		mask_feature = [tf_mask.astype(np.bool).tobytes()]

		max_examples = all_inputs.shape[2]

		# get accessibility path
		joined_accessibility_filename = ''
		for file in os.listdir(self._DNASE_BED_DIRNAME):
			if fnmatch.fnmatch(file, ('*%s*%s*' % (cell, self._JOINED_PY_BED_EXTENSION))):
					tf.logging.info("Found joined bed file for cell type %s" % (cell))
					joined_accessibility_filename = self._DNASE_BED_DIRNAME + "/" + file
					accessibility_data = pybedtools.BedTool(joined_accessibility_filename)
                               
                               
		# if file DNE, create it
		if (joined_accessibility_filename == ''):
			tf.logging.info("No joined accessibility data for cell type %s. Joining accessibility and positions..." % (cell))
			# get accessibility path
			accessibility_filename = ''
			for file in os.listdir(self._DNASE_BED_DIRNAME ):
					if fnmatch.fnmatch(file, ('*%s*' % cell)):
						accessibility_filename = self._DNASE_BED_DIRNAME  + "/" + file
			tf.logging.info("Joining with %s..." % (accessibility_filename))                               
			if (accessibility_filename != ''):
				joined_accessibility_filename, accessibility_data = save_merged_bedfile(tmp_dir + '/' + self._DEEPSEA_GENOME_REGIONS_FILENAME, accessibility_filename, self._JOINED_PY_BED_EXTENSION)
				tf.logging.info("Successfully joined and saved to %s..." % (joined_accessibility_filename))                               


		for i, bed_row_i in enumerate(range(start, stop)):
			if (i % 1000 == 0):
				tf.logging.info("Completed %d records for cell type %s" % (i, cell))
            
			if (i >= max_examples):
				return
            
			# read bed file line to get chr, start and end. 
			# Increment by one because first line of file is blank
			bed_row = linecache.getline(os.path.join(tmp_dir, self._DEEPSEA_GENOME_REGIONS_FILENAME), bed_row_i+1).split('\t')
			region_chr = bed_row[0]
			region_start = int(bed_row[1]) - 400 # Get flanking base pairs. Deepsea only looks at middle 200 bp
			region_stop = int(bed_row[2]) + 400  # Get flanking base pairs. Deepsea only looks at middle 200 bp

            
			# inputs is 1000 * 6:
			# input1: The first four are one-hot bases,
			# input2: The fifth is DNAse (the same value for every base) OR based on accessibility file, if present
			# input3: The sixth is strand
			inputs1 = all_inputs[:, :, i] # 1000 x 4

			if joined_accessibility_filename == '':
				tf.logging.info("Warning: no accessibility data for cell type %s. Putting in DNase DeepSea labels as accessibility features..." % (cell))
				inputs2 = np.array([[all_targets[dnase_dict[cell], i]]] * 1000)
			else:
				inputs2 = get_accessibility_vector_pybed(i, accessibility_data)
                
			inputs3 = np.array([[0 if i < self.num_examples / 2 else 1]] * 1000) # 1000 x 1
			inputs = np.concatenate([inputs1, inputs2, inputs3], 1) # final result is 1000 * 6
            
			# y is queried from the target matrix. We mask by whether we
			# actually have this TF for this cell type.
			targets = np.array([all_targets[c, i] for c in tf_locs]) * tf_mask
            
			yield {
				'chr':    [region_chr],
				'start':  [region_start],
				'stop':   [region_stop],
				'inputs/data': [inputs.astype(np.int32).tobytes()],
				'targets': [targets.astype(np.bool).tobytes()],
				'mask': mask_feature
			}

	def parse_feature_name(self, path):
		''' 
		:param path filepath to file containing feature labels for targets array
		:return dnase_dict dictionary of cell type and corresponding position in the targets array
		:return tf_dict dictionary of all proteins for all cell types, and their corresponding position in the targets array
		'''
    
		with open(path) as f:
			for _ in f:
				break

			i = 0
			dnase_dict = {}
			tf_dict = {}

			for line in f:
				if i == 815:
					break
				_, label = line.split('\t')
				cell, assay, note = label.split('|')
				if assay == "DNase":
					dnase_dict[cell] = i
				else:
					if cell not in tf_dict:
						tf_dict[cell] = {}
					if assay not in tf_dict[cell]:
						tf_dict[cell][assay] = i
				i += 1
		return dnase_dict, tf_dict

	def get_feature_indices(self, tf_dict, cell):
		'''
		Builds a mask that masks proteins based on the current cell type. If the protein does not exist for this cell, mask it.
		:param tf_dict dictionary of proteins and positions in target array for all cells
		:param cell cell to mask
		:return tf_locs
		:return tf_mask 0/1 vector to mask proteins
		'''
        
		tf_vec = []
		i = 0
		for train_protein in self.train_proteins:
			if train_protein in tf_dict[cell]:
				tf_vec += [(tf_dict[cell][train_protein], 1)]
			else:
				tf_vec += [(0, 0)]
		tf_locs = np.array([v[0] for v in tf_vec])
		tf_mask = np.array([v[1] for v in tf_vec])
		return tf_locs, tf_mask
