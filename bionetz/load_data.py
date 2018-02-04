import h5py
import numpy as np
from scipy.io import loadmat

def test_and_valid_batches(batch_size, input_, target, seperate_dnase = False):
    while True:
        for i in range(int(input_.shape[0]/batch_size)):
            if seperate_dnase:
                yield (input_[i*batch_size:(i+1)*batch_size,:,0:1000].transpose([0,2,1]), 
                        target[i*batch_size:(i+1)*batch_size,:126],
                        target[i*batch_size:(i+1)*batch_size,126:])
            else:
                yield (input_[i*batch_size:(i+1)*batch_size,:,0:1000].transpose([0,2,1]),
                       np.zeros([batch_size, 126]),
                       target[i*batch_size:(i+1)*batch_size, 126:])

def train_batches(batch_size, input_, target, seperate_dnase = False):
    while True:
        num_samples = input_.shape[2]
        num_batches = num_samples / batch_size
        batch_order = np.random.permutation(int(num_batches))
        for i in batch_order:
            if seperate_dnase:
                yield (input_[0:1000,:,i*batch_size:(i+1)*batch_size].transpose([2, 0, 1]),
                        target[:126,i*batch_size:(i+1)*batch_size].transpose([1, 0]),
                        target[126:,i*batch_size:(i+1)*batch_size].transpose([1, 0]))
            else:
                yield (input_[0:1000,:,i*batch_size:(i+1)*batch_size].transpose([2, 0, 1]),
                       np.zeros([batch_size, 126]),
                       target[126:,i*batch_size:(i+1)*batch_size].transpose([1, 0]))


# Yields (
#  (batchsize * 4 * 1000) - Sequence for X
#  (batchsize * 126) - DNAse for X, or all 0's
#  (batchsize * (919 - 126 * seperate_dnase)) - Y, optionally including DNAse
# )
def make_data_iterator(path, batch_size, seperate_dnase = False):
    if 'train.mat' == path[-9:]:
        tmp = h5py.File(path)

        i = tmp['trainxdata']
        t = tmp['traindata']

        return train_batches(batch_size, i, t, seperate_dnase)

    elif 'valid.mat' == path[-9:]:
        tmp = loadmat(path)
        i = tmp['validxdata']
        t = tmp['validdata']
        return test_and_valid_batches(batch_size, i, t, seperate_dnase)

