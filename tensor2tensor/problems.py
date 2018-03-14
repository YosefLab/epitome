from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import multiprocessing as mp
import os
import zipfile


# Dependency imports

import h5py
import numpy as np
from scipy.io import loadmat


from tensor2tensor.data_generators import dna_encoder
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import metrics
from tensor2tensor.utils import registry

import tensorflow as tf

		
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
		return 2

	@property
	def num_shards(self):
		"""Number of TF-Record shards."""
		return 100

	def generator(self, tmp_dir, is_training):
		raise NotImplementedError()
		# _get_deepsea(tmp_dir)

		# if is_training:
		# 	return _deepsea_train_generator(tmp_dir, cell, start, stop)
		# else:
		# 	return _deepsea_test_generator(tmp_dir, cell, start, stop)

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


class DeepSeaProblem(ProteinBindingProblem):
	"""Transcription factor binding site prediction."""
	_DEEPSEA_DOWNLOAD_URL = ("http://deepsea.princeton.edu/media/code/"
	                     "deepsea_train_bundle.v0.9.tar.gz")
	_DEEPSEA_FILENAME = "deepsea_train_bundle.v0.9.tar.gz"
	_DEEPSEA_DIRNAME = "deepsea_train"
	_DEEPSEA_TRAIN_FILENAME = "deepsea_train/train.mat"
	_DEEPSEA_TEST_FILENAME = "deepsea_train/valid.mat"
	_DEEPSEA_FEATURES_FILENAME = ""

	def generator(self, tmp_dir, is_training):
		self._get_data(tmp_dir)

		if is_training:
			return self._train_generator(tmp_dir)
		else:
			return self._test_generator(tmp_dir)

	def _get_data(self, directory):
		"""Download all Deepsea files to directory unless they are there."""
		zip_name = os.path.join(directory, self._DEEPSEA_FILENAME)
		generator_utils.maybe_download(
			directory, self._DEEPSEA_FILENAME, self._DEEPSEA_DOWNLOAD_URL)
		zipfile.ZipFile(zip_name, "r").extractall(zip_name)

	def _train_generator(self, tmp_dir):
		tmp = h5py.File(os.path.join(tmp_dir, self._DEEPSEA_TRAIN_FILENAME))
        all_inputs, all_targets = tmp['trainxdata'], tmp['traindata']

        for i in range(all_inputs.shape[2]):
        	inputs = all_inputs[:, :, i].transpose([2, 0, 1])
        	targets = all_targets[:, i].transpose([1, 0])

        	yield {
        		"inputs": [inputs.astype(np.bool).tobytes()],
        		"targets": [targets.astype(np.bool).tobytes()],
        	}

	def _test_generator(tmp_dir, cell, start, stop):
        tmp = loadmat(path)
        all_inputs, all_targets = tmp['validxdata'], tmp['validdata']

        for i in range(all_inputs.shape[0]):
        	inputs = all_inputs[i].transpose([0, 2, 1])
        	targets = all_targets[i].transpose([1, 0])

        	yield {
        		"inputs": [inputs.astype(np.bool).tobytes()],
        		"targets": [targets.astype(np.bool).tobytes()],
        	}

    def example_reading_spec(self):
	    data_fields = {
	        "inputs": tf.FixedLenFeature([], tf.string),
	        "targets": tf.FixedLenFeature([], tf.string),
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

