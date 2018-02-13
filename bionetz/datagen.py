"""
Converst the deepsea data into tfrecords.
REDACTED: we no longer use this file.
"""

import os
import argparse
import threading
import h5py
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
import iio

tmp = h5py.File('./deepsea_train/train.mat', 'r')
train_input = tmp['trainxdata']
train_target = tmp['traindata']

tmp = loadmat('./deepsea_train/valid.mat')
valid_input = tmp['validxdata']
valid_target = tmp['validdata']

def test_and_valid_iter(input_, target):
    for i in xrange(input_.shape[0]):
        yield iio.make_example({
            'input': iio.make_float_feature(input_[i].flatten()),
            'target': iio.make_int64_feature(target[i].flatten()),
        })
    
def train_iter(input_, target):
    for i in xrange(input_.shape[2]):
        yield iio.make_example({
            'input': iio.make_float_feature(input_[:,:,i].flatten()),
            'target': iio.make_int64_feature(target[:,i].flatten()),
        })

def write_tfrecords(examples, filename, num_shards=100):
    filenames = iio.sharded_filenames(filename, num_shards)
    writers = map(tf.python_io.TFRecordWriter, filenames)
    for i, example in enumerate(examples):
        t = threading.Thread(
            target=writers[i % num_shards].write,
            args=(example.SerializeToString(),))
        t.daemon = True
        t.start()

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--num_shards', '-n', type=int, default=100)
	parser.add_argument('--output_dir', '-o', type=str, default='./output/')
	args = parser.parse_args()

	if not os.path.exists(args.output_dir):
		os.mkdir(args.output_dir)

	print('Writing validation examples...')
	write_tfrecords(
	    examples=test_and_valid_iter(valid_input, valid_target),
	    filename=os.path.join(args.output_dir, 'valid.tfrecord'),
	    num_shards=args.num_shards)
	print('Done.')

	print('Writing training examples...')
	write_tfrecords(
	    examples=train_iter(train_input, train_target),
	    filename=os.path.join(args.output_dir, 'train.tfrecord'),
	    num_shards=args.num_shards)
	print('Done.')

if __name__ == '__main__':
	main()
