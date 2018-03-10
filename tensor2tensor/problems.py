from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import multiprocessing as mp
import os

# Dependency imports

import h5py
import numpy as np

from tensor2tensor.data_generators import dna_encoder
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import metrics
from tensor2tensor.utils import registry

import tensorflow as tf

_DEEPSEA_DOWNLOAD_URL = ("http://deepsea.princeton.edu/media/code/"
	                     "deepsea_train_bundle.v0.9.tar.gz")
_DEEPSEA_FILENAME = "deepsea_train_bundle.v0.9.tar.gz"
_DEEPSEA_TRAIN_FILENAME = ""
_DEEPSEA_TEST_FILENAME = ""
_DEEPSEA_FEATURES_FILENAME = ""


def _get_deepsea(directory):
	"""Download all Omniglot files to directory unless they are there."""
	generator_utils.maybe_download(
		directory, _DEEPSEA_FILENAME, _DEEPSEA_DOWNLOAD_URL)

	##################################################
	# Your code here!
	# Any aditional preprocessings such as unzipping
	##################################################


def _deepsea_train_generator(tmp_dir, cell, start, stop):
	for i in range(100):

		##################################################
		# Your code here!
		# Reading from an HDF5 file
		##################################################

		return inputs, targets


def _deepsea_test_generator(tmp_dir, cell, start, stop):
	for i in range(100):

		##################################################
		# Your code here!
		# Reading from a MatLab file
		##################################################

		return inputs, outputs


def _common_generator(tmp_dir, cell, start, stop, is_training):
	if is_training:
		return _deepsea_train_generator(tmp_dir, cell, start, stop)
	else:
		return _deepsea_test_generator(tmp_dir, cell, start, stop)
		

class TranscriptionFactorProblem(problem.Problem):
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
		return 5

	@property
	def num_classes(self):
		"""Binary classification."""
		return 2

	@property
	def num_shards(self):
		"""Number of TF-Record shards."""
		return 100

	@property
	def train_cells(self):
		return ["HeLa-S3", "GM12878", "H1-hESC", "HepG2"]  # Fix me!

	@property
	def test_cells(self):
		return ["K562"]  # Fix me!

	@property
	def train_start(self):
		return 0  # Fix me!

	@property
	def train_stop(self):
		return 1000  # Fix me!

	@property
	def test_start(self):
		return 0  # Fix me!

	@property
	def test_stop(self):
		return 1000  # Fix me!

	def generator(self, tmp_dir, is_training):
		_get_deepsea(tmp_dir)

		cells = self.train_cells if is_training else self.test_cells
		start = self.train_start if is_training else self.test_start
		stop = self.train_stop if is_training else self.test_stop

		for cell in cells:
			for inputs, targets in _common_generator(
				tmp_dir, cell, start, stop, is_training):
				yield inputs, targets

	def generate_data(self, data_dir, tmp_dir, task_id=-1):
		generator_utils.generate_dataset_and_shuffle(
			self.generator(tmp_dir, is_training=True),
			self.training_filepaths(data_dir, self.num_shards, shuffled=False),
			self.generator(tmp_dir, is_training=False),
			self.dev_filepaths(data_dir, 1, shuffled=False))

	def hparams(self, defaults, unused_model_hparams):
		p = defaults
		p.input_modality = {"inputs": (registry.Modalities.GENERIC, None)}
		p.target_modality = (registry.Modalities.SYMBOL, 2)
		p.input_space_id = problem.SpaceID.GENERIC
		p.target_space_id = problem.SpaceID.GENERIC

	def example_reading_spec(self):
		# Do not modify! Parse bytestrings in preprocess_example.
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
	    inputs = tf.reshape(tf.decode_raw(inputs, tf.float32), inputs_shape)
	    targets = tf.reshape(tf.decode_raw(targets, tf.bool), targets_shape)

	    ################################################
	    # Your code here!  Any aditional preprocessing
	    # For example, converting to floats and ints.
	    ################################################
	    
	    example["inputs"] = tf.to_float(inputs)
	    example["targets"] = tf.to_int32(targets)

    	return example

	def eval_metrics(self):
		################################################
		# Get the metrics you want to use.
		# For example, we might want AUC, FPR, TPR, etc.
		################################################
    	return [metrics.Metrics.LOG_POISSON, metrics.Metrics.R2]
