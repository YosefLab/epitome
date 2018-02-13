"""
Main entry point for training.
Trains a network that predicts TF binding sites.
"""

import os
import argparse

print("Importing TensorFlow...")
import tensorflow as tf
print("Done importing TensorFlow!")

import logz
from models import build_CNN_graph
from train import train
from load_data import make_data_iterator


def main():
	# TODO(weston): Add help text for arguments.
	parser = argparse.ArgumentParser()
	parser.add_argument('--train', default = '../../deepsea_train/train.mat')
	parser.add_argument('--valid', default = '../../deepsea_train/valid.mat')
	parser.add_argument('--DNAse', action='store_true')
	parser.add_argument('--batch', default = 64)
	parser.add_argument('--rate', default = 1e-3)
	parser.add_argument('--pos_weight', default = 50)
	parser.add_argument('--name', default = 'test')
	parser.add_argument('--logdir', default = '../logs')
	parser.add_argument('--iterations', default = int(3e6))
	parser.add_argument('--log_freq', default = 1000)
	parser.add_argument('--save_freq', default = 20000)
	args =  parser.parse_args()

	# Configure the logging and checkpointing directories.
	logdir = os.path.join(args.logdir, args.name)
	save_path = os.path.join(logdir, "model.ckpt")
	logz.configure_output_dir(logdir)

	# This builds the tf graph, and returns a dictionary of the ops needed for 
	# training and testing.
	ops = build_CNN_graph(DNAse=args.DNAse, pos_weight=float(args.pos_weight),
		                  rate=float(args.rate))

	# This function contains the training and validation loops.
	train_iterator = make_data_iterator(args.train, args.batch, args.DNAse) 
	valid_iterator = make_data_iterator(args.valid, args.batch, args.DNAse)

	# Train the network.
	train(ops, int(args.log_freq), int(args.save_freq), save_path, args.DNAse,
	     int(args.iterations), train_iterator, valid_iterator)


if __name__ == '__main__':
	main()