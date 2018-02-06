#
# Licensed to Big Data Genomics (BDG) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The BDG licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


import numpy as np
import h5py
from scipy.io import loadmat


################################# Define Networks ################################

def chunk_iterator(array, chunk_size=1):
    # iterates and chunks along first dimension
    for i in range(array.shape[0] // chunk_size):
        yield array[chunk_size * i: (i + 1) * chunk_size]


def train_iterator(source, batch_size=32, num_epochs=200):
    # load a h5py file (dictionary*) of datasets (arrays*)
    tmp = h5py.File(source, 'r')
    inputs = tmp['trainxdata']
    targets = tmp['traindata']
    # last index in the batch dimension
    num_examples = inputs.shape[2]
    for epoch in range(num_epochs):
        permutation = np.random.permutation(num_examples)
        for indices in chunk_iterator(permutation, batch_size):
            # h5py datasets require sorted point-wise indexing
            inputs_batch = inputs[:,:,sorted(indices)].transpose([2, 0, 1])
            targets_batch = targets[:,sorted(indices)].transpose([1, 0])
            yield inputs_batch, targets_batch


def valid_iterator(source, batch_size=32, num_epochs=200):
    # load a dictionary of arrays
    tmp = loadmat(source)
    inputs = tmp['validxdata']
    targets = tmp['validdata']
    # first index is the batch dimension
    num_examples = inputs.shape[0]
    for epoch in range(num_epochs):
        permutation = np.random.permutation(num_examples)
        for indices in chunk_iterator(permutation, batch_size):
            yield inputs[indices,:,:], targets[indices,:]
