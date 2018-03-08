"""
Main entry point for training.
Trains a network that predicts TF binding sites.
"""

import os
import argparse
import pickle

print("Importing TensorFlow...")
import tensorflow as tf
print("Done importing TensorFlow!")

import logz
from models import build_cnn_graph, cnn_hp, load_hparams, save_hparams, parse_hparams_string
from train import train
from load_data import make_data_iterator


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--train', default='../../deepsea_train/train.mat',
	 help='path to file containing training data. Must end in "train.mat"')
	parser.add_argument('--valid', default='../../deepsea_train/valid.mat',
	 help='path to file containing validation data. Must end in "valid.mat"')
	parser.add_argument('--valid_size', default=1000,
	 help='the number of examples in the validation set (needed since we cant' +
	 'count tf_records')
	parser.add_argument('--DNAse', action='store_true',
	 help='use DNAse for classification')
	parser.add_argument('--batch', default=64,
	 help='the training batch size')
	parser.add_argument('--rate', default=1e-3,
	 help='the learning rate for training')
	parser.add_argument('--pos_weight', default=50,
	 help='the amount by which positive examples are wieghted')
	parser.add_argument('--name', default='test',
	 help='the name of the model to be created or loaded')
	parser.add_argument('--logdir', default='../logs',
	 help='the directory to save logs in')
	parser.add_argument('--custom_hparams', default=None,
	 help='custom comma-separated string of hparams to use for training. For'
	 ' example, `n_conv_layers=10,gated=True`.')
	parser.add_argument('--iterations', default=int(3e6),
	 help='the number of batches to train on')
	parser.add_argument('--log_freq', default=1000,
	 help='the frequency, in batches, at which results are logged during' + 
	 'training')
	parser.add_argument('--save_freq', default=20000,
	 help='the frequency, in batches, at which results are saved during' + 
	 'training')
	parser.add_argument('--seed', default=1,
	 help='the random seed to be fed into tensorflow')
	parser.add_argument('--tfrecords', action='store_true',
	 help='read data in from tfrecords')
	args = parser.parse_args()

	# Configure the logging and checkpointing directories.
	tf.set_random_seed(args.seed)

	logdir = os.path.join(args.logdir, args.name)
	save_path = os.path.join(logdir, "model.ckpt")
	hp_path = os.path.join(logdir, "model.hp")
	logz.configure_output_dir(logdir)

	hps = None
	# TODO(weston): Figure out why this doesn't work
	# if os.path.isdir(logdir):
	# 	hps = load_hparams(hp_path)
	# 	if hps:
	# 		print("Model restored.")
	if not hps:
		custom_kwargs = parse_hparams_string(args.custom_hparams)
		hps = cnn_hp(**custom_kwargs)
		print("!", hps.to_json())
		save_hparams(hp_path, hps)	
		print("Model initialized.")


	if args.tfrecords:
		num_logits = 18
	else:
		num_logits = 815-125

	# This builds the tf graph, and returns a dictionary of the ops needed for 
	# training and testing.
	ops = build_cnn_graph(DNAse=args.DNAse, pos_weight=float(args.pos_weight),
		                  tfrecords=args.tfrecords,
		                  num_logits=num_logits, hp=hps)

	# This function contains the training and validation loops.
	train_iterator=make_data_iterator(args.train, args.batch, args.DNAse, 
		tfrecords=args.tfrecords) 
	valid_iterator=make_data_iterator(args.valid, args.batch, args.DNAse, 
		tfrecords=args.tfrecords, shuffle=False) 

	# Train the network.
	train(ops, int(args.log_freq), int(args.save_freq), save_path, args.DNAse,
	     int(args.iterations), train_iterator, valid_iterator, 
	     num_logits=num_logits, tfrecords=args.tfrecords, rate=float(args.rate),
	     valid_size=(int(args.valid_size)//int(args.batch)+1))




if __name__ == '__main__':
	main()
