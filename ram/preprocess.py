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

import os
import re
import glob
import numpy as np
import h5py
import argparse


################################# Utility Functions ################################

# possible bases, where 'N' is the unknown base
BASES = list('ACTGN')

################################# Utility Functions ################################

def parse_dense_vector_string(string, dtype=None):
    # search for the DenseVector string wrapper
    match = re.search('DenseVector\((.+?)\)', string)
    if match is None:
        # if there is no DenseVector wrapper then
        # the string is just comma separated numbers
        # and turn it into an array
        array = np.asarray(map(eval, string.split(',')))
    else:
        # extract content of regex match, if there is a match
        # and turn it into an array
        array = np.asarray(map(eval, match.group(1).split(', ')))
    # returns an array, maybe enforcing a data type
    return array if dtype is None else array.astype(dtype)


def seq_to_one_hot(string):
    # mapping bases to indices
    indices = map(lambda base: BASES.index(base), list(string))
    # using indices to index an idenity matrix
    # equivalent to one-hot-encoding the data
    return np.eye(len(BASES))[indices]


################################# Main Body ################################

def main():
    parser = argparse.ArgumentParser()

    # the input file to be processed (.txt)
    parser.add_argument('--infile', '-i', type=str)
    # the output file to be written to (.hdf5)
    parser.add_argument('--outfile', '-o', type=str)
    # how often to log the preprocessing status
    parser.add_argument('--log_freq', '-l', type=int, default=1000)
    # generate an hdf5 file?
    parser.add_argument('--hdf5', type=bool, default=False)
    args = parser.parse_args()

    # check the success flag
    assert '_SUCCESS' in os.listdir(args.infile)
    # getting shardnames, adam formatting
    shards = glob.glob(os.path.join(args.infile, 'part-*'))
    # number of training examples in the infile
    # code below counts the number of lines
    n = sum(sum(1 for _ in open(x)) for x in shards)

    if args.hdf5:
        # create a hdf5 file to write to
        f = h5py.File(args.outfile, "w")
        f.create_dataset("label", (n, 1), maxshape=(n, 1), dtype='i')
        f.create_dataset("atac", (n, 1000), maxshape=(n, 1000), dtype='i')
        f.create_dataset("seq", (n, 1000, len(BASES)),
                         maxshape=(n, 1000, len(BASES)), dtype='i')
    else:
        # use a dictionary of lists for now
        f = {}
        f['label'] = np.zeros([n, 1], dtype=int)
        f['atac'] = np.zeros([n, 1000], dtype=int)
        f['seq'] = np.zeros([n, 1000, len(BASES)], dtype=int)

    # index of the example being processed
    i = 0

    for shard in shards:
        for line in open(shard):
            # each field is delimited by a semi-colon
            label, atac, seq = line.strip().split(';')

            # parse each field
            label = parse_dense_vector_string(label)
            atac = parse_dense_vector_string(atac)
            seq = seq_to_one_hot(seq)

            # index the datasets in the hdf5 file
            try:
                f['label'][i] = label
                f['atac'][i] = atac
                f['seq'][i] = seq
                i += 1
            except:
                # filter out bad examples
                n -= 1
                continue

            # status update for logging
            if i % args.log_freq == 0:
                print('[ %s / %s ] :: %.4f percent' % (
                    str(i).rjust(6, '0'),
                    str(n).rjust(6, '0'),
                    (float(i) / n) * 100))

    if args.hdf5:
        # resize to the number of good examples
        f['label'].resize(i + 1, axis=0)
        f['atac'].resize(i + 1, axis=0)
        f['seq'].resize(i + 1, axis=0)
    else:
        # truncate by slicing and save to outfile
        f = {k: v[:i + 1] for k, v in f.iteritems()}
        np.savez(args.outfile, **f)

    # print for debugging
    print('%i examples written to %s' % (i + 1, args.outfile))


if __name__ == "__main__":
    main()
