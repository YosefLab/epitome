print("Importing packages")
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import argparse
print("Finished importing packages")

import logz
from models import build_CNN_graph
from train import train
from load_data import make_data_iterator

# TODO: Add help text for arguments
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

logdir = os.path.join(args.logdir, args.name)
save_path = os.path.join(logdir, "model.ckpt")
logz.configure_output_dir(logdir)

# This builds the tf graph, and returns a dictionary of the ops needed for 
# training and testing.
ops = build_CNN_graph(DNAse = args.DNAse,
                    pos_weight = args.pos_weight,
                    rate = args.rate)

# This function contains the training and validation loops.
train_iterator = make_data_iterator(args.train, args.batch, args.DNAse) 
valid_iterator = make_data_iterator(args.valid, args.batch, args.DNAse)
train(ops, args.log_freq, args.save_freq, save_path, args.DNAse,
     args.iterations, train_iterator, valid_iterator)

