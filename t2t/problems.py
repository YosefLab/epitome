import math
import multiprocessing as mp
import os
import tarfile

# Dependency imports

import h5py
import numpy as np
from scipy.io import loadmat

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

	@property
	def train_cells(self):
		return ['HeLa-S3', 'GM12878', 'H1-hESC', 'HepG2', 'K562']

	@property
	def train_proteins(self):
		return ['p300', 'NRSF', 'CTCF', 'GABP', 'JunD', 'CEBPB', 'Pol2', 'EZH2',
				'Mxi1', 'Max', 'RFX5', 'TAF1', 'Nrf1', 'Rad21', 'TBP', 'USF2',
				'c-Myc','CHD2']

	@property
	def num_examples(self):
		return 4400000

	@property
	def num_output_predictions(self):
		"""Number of binding site predictions."""
		return len(self.train_proteins)

	@property
	def num_channels(self):
		"""Input channels."""
		return 6

	def _get_data(self, directory):
		"""Download all files to directory unless they are there."""
		generator_utils.maybe_download(
			directory, self._DEEPSEA_FILENAME, self._DEEPSEA_DOWNLOAD_URL)
		tar_name = os.path.join(directory, self._DEEPSEA_FILENAME)
		tar = tarfile.open(tar_name, "r:gz")
		tar.extractall(directory)
		tar.close()

	def generator(self, tmp_dir, is_training):
		self._get_data(tmp_dir)

		tmp = h5py.File(args.data)
		all_inputs, all_targets = tmp['trainxdata'], tmp['traindata']

		if is_training:
			for cell in self.train_cells:
				for example in cell_generator(all_inputs, all_targets, cell, 0, 
									self.num_examples * .9):
					yield example
		else:
			for cell in self.train_cells:
				for example in cell_generator(all_inputs, all_targets, cell, 
									self.num_examples * .9, self.num_examples):
					yield example

	def example_reading_spec(self):
		data_fields = {
			"inputs/data": tf.FixedLenFeature([], tf.string),
			"inputs/shape": tf.FixedLenFeature([], tf.string),
			"targets": tf.FixedLenFeature([], tf.string),
			"mask": tf.FixedLenFeature([], tf.string)
		}
		data_items_to_decoders = None
		return (data_fields, data_items_to_decoders)

	def preprocess_example(self, example, mode, unused_hparams):
		del mode

		inputs = example["inputs/data"]
		inputs_shape = example["inputs/shape"]
		targets = example["targets"]
		mask = example["mask"]

		targets_shape = [self.num_output_predictions, 1]

		# Parse the bytestring based on how you encoded it in common_generator
		inputs = tf.reshape(tf.decode_raw(inputs, tf.bool), 
							tf.decode_raw(inputs_shape, tf.int32))
		targets = tf.reshape(tf.decode_raw(targets, tf.bool), targets_shape)
		mask = tf.reshape(tf.decode_raw(mask, tf.bool), targets_shape)

		example["inputs"] = tf.to_float(inputs)
		example["targets"] = tf.to_int32(targets)
		example["mask"] = tf.to_int32(mask)

		return example

	def cell_generator(all_inputs, all_targets, cell, start, stop):
		# Builds dicts of indicies of different features
		dnase_dict, tf_dict = self.parse_feature_name(self._DEEPSEA_FEATURES_FILENAME)
		
		# builds the vector of locations for querying from matrix, and the mask
		tf_locs, tf_mask = self.get_feature_indices()

		# Pre-build these features
		mask_feature = [tf_mask.astype(np.bool).tobytes()]

		num_samples = all_inputs.shape[2]
		for i in range(start, min(stop, num_samples)):

			# x is 1000 * 6:
			# The first four are one-hot bases,
			# The fifth is DNAse (the same value for every base)
			# The sixth is strand
			inputs1 = all_inputs[:, :, i]
			inputs2 = np.array([[all_targets[dnase_dict[cell], i]]] * 1000)
			inputs3 = np.array([[0 if i < self.num_examples / 2 else 1]] * 1000)
			inputs = np.concatenate([inputs1, inputs2, inputs3], 1)

			# y is queried from the target matrix. We mask by whether we 
			# actually have this TF for this cell type.
			targets = np.array([all_targets[c, i] for c in tf_locs]) * tf_mask

			yield {
				'inputs/data': [inputs.flatten().astype(np.bool).tobytes()],
				'inputs/shape': [inputs.shape.astype(np.int32).tobytes()],
				'targets': [targets.astype(np.bool).tobytes()],
				'mask': mask_feature
			}

	def parse_feature_name(self, path):
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

	def get_feature_indices(self):
		tf_vec = []
		i = 0
		for tf in tfs:
			if tf in tf_dict[cell]:
				tf_vec += [(tf_dict[cell][tf], 1)]
			else:
				tf_vec += [(0, 0)]
		tf_locs = np.array([v[0] for v in tf_vec])
		tf_mask = np.array([v[1] for v in tf_vec])
		return tf_locs, tf_mask
