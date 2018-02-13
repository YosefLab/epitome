"""
Loads in deepsea data to train on.
"""

import numpy as np

import h5py
from scipy.io import loadmat


def test_and_valid_batches(batch_size, input_, target, seperate_dnase=False):
    """Creates training batches for test and validation formatted npz files.

    Loops through one epoch of the data.

    Args:
        batch_size: Int. The batch size to yield.
        input_: Input array with shape [num_examples, 4, 1000].
        target: Target array with shape [num_examples, 919].
        seperate_dnase: Boolean. Whether to separate DNase from TF binding
            and histone markers in the targets array.

    Yields:
        A 2-tuple with an input batch with shape [batch_size, 1000, 4] and a
        target batch with shape [batch_size, 919]. If seperate_dnase is True,
        then you get a 3-tuple with an input batch with shape
        [batch_size, 1000, 4], dnase batch with shape [batch_size, 126] and a
        TF binding and histone batch with shape [batch_size, 793].
    """
    while True:
        for i in range(int(input_.shape[0]/batch_size)):
            if seperate_dnase:
                yield (input_[i*batch_size:(i+1)*batch_size,:,0:1000
                              ].transpose([0,2,1]), 
                        target[i*batch_size:(i+1)*batch_size,:126],
                        target[i*batch_size:(i+1)*batch_size,126:])
            else:
                yield (input_[i*batch_size:(i+1)*batch_size,:,0:1000
                              ].transpose([0,2,1]),
                       np.zeros([batch_size, 126]),
                       target[i*batch_size:(i+1)*batch_size, 126:])


def train_batches(batch_size, input_, target, seperate_dnase=False):
    """Creates training batches for train formatted h5py files.

    Loops through one epoch of the data.

    Args:
        batch_size: Int. The batch size to yield.
        input_: Input h5py with shape [1000, 4, num_examples].
            NOTE: notice this is inconsistent with train/valid.
        target: Target array with shape [919, num_examples].
            NOTE: notice this is inconsistent with train/valid.
        seperate_dnase: Boolean. Whether to separate DNase from TF binding
            and histone markers in the targets array.

    Yields:
        A 2-tuple with an input batch with shape [batch_size, 1000, 4] and a
        target batch with shape [batch_size, 919]. If seperate_dnase is True,
        then you get a 3-tuple with an input batch with shape
        [batch_size, 1000, 4], dnase batch with shape [batch_size, 126] and a
        TF binding and histone batch with shape [batch_size, 793].
    """
    while True:
        num_samples = input_.shape[2]
        num_batches = num_samples / batch_size
        batch_order = np.random.permutation(int(num_batches))
        for i in batch_order:
            if seperate_dnase:
                yield (input_[0:1000,:,i*batch_size:(i+1)*batch_size
                              ].transpose([2, 0, 1]),
                        target[:126,i*batch_size:(i+1)*batch_size
                               ].transpose([1, 0]),
                        target[126:,i*batch_size:(i+1)*batch_size
                               ].transpose([1, 0]))
            else:
                yield (input_[0:1000,:,i*batch_size:(i+1)*batch_size
                              ].transpose([2, 0, 1]),
                       np.zeros([batch_size, 126]),
                       target[126:,i*batch_size:(i+1)*batch_size
                              ].transpose([1, 0]))


def repeater(iterator, num_repeat=None):
    """Repeats an iterator over and over.

    Args:
        iterator: An iterator.
        num_repeat: None or Int. Number of times to repeat. If this is None
            the iterator will loop 9999 times.

    Yields:
        Elements in the iterator over and over.
    """
    if num_repeat is None:
        num_repeat = 9999

    for _ in range(num_repeat):
        for item in iterator:
            yield item


def make_data_iterator(path, batch_size, seperate_dnase=False, num_repeat=None):
    """Makes a deepsea data iterator from a path.

    Args:
        path: String. Path to the data file. This should be formatted in the
            same way deepsea has. And should be named `train.mat`, `valid.mat`,
            or `test.mat`.
        batch_size: Int. The batch size to yield.
        seperate_dnase: Boolean. Whether to separate DNase from TF binding
            and histone markers in the targets array.
        num_repeat: None or Int. Number of times to repeat. If this is None
            the iterator will loop 9999 times.

    Yields:
        A 2-tuple with an input batch with shape [batch_size, 1000, 4] and a
        target batch with shape [batch_size, 919]. If seperate_dnase is True,
        then you get a 3-tuple with an input batch with shape
        [batch_size, 1000, 4], dnase batch with shape [batch_size, 126] and a
        TF binding and histone batch with shape [batch_size, 793].
    """
    if path.endswith('train.mat'):
        # Read an hdf5 file.
        tmp = h5py.File(path)
        i, t = tmp['trainxdata'], tmp['traindata']
        iterator = train_batches(batch_size, i, t, seperate_dnase)
        return repeater(iterator, num_repeat)
    elif path.endswith('valid.mat') or path.endswith('test.mat'):
        # Read a matlab file.
        tmp = loadmat(path)
        i, t = tmp['validxdata'], tmp['validdata']
        iterator = test_and_valid_batches(batch_size, i, t, seperate_dnase)
        return repeater(iterator, num_repeat)
    else:
        raise NotImplementedError('Unrecognized file: %s' % path)