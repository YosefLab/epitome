# TODO(weston): clean up this file.

import tensorflow as tf
import numpy as np
import h5py
import iio
import os
import argparse
import glob



# Builds dictionaries to look up coordinates
def parse_feature_name(path):
    with open(path) as f:
        for _ in f:
            break

        i = 0
        dnase_dict = {}
        tf_dict = {}

        for line in f:
            if i == 815:
                break
            _, label = line.split('\t')
            cell, assay, note = label.split('|')
            if assay == "DNase":
                dnase_dict[cell] = i
            else:
                if cell not in tf_dict:
                    tf_dict[cell] = {}
                if assay not in tf_dict[cell]:
                    tf_dict[cell][assay] = i
            i += 1
    return dnase_dict, tf_dict

cells = ['HeLa-S3', 'GM12878', 'H1-hESC', 'HepG2', 'K562']
tfs = ['p300', 'NRSF', 'CTCF', 'GABP', 'JunD', 'CEBPB', 'Pol2', 'EZH2',
        'Mxi1', 'Max', 'RFX5', 'TAF1', 'Nrf1', 'Rad21', 'TBP', 'USF2',
             'c-Myc','CHD2']

# Generates records for a given cell type and range
def records_iterator(input_, target, cell, start, stop):
    # Builds dicts of indicies of different features
    dnase_dict, tf_dict = parse_feature_name(
    "../../DeepSEA-v0.94/resources/feature_name")
    
    # builds the vector of locations, and the mask
    tf_vec = []
    i = 0
    for tf in tfs:
        if tf in tf_dict[cell]:
            tf_vec += [(tf_dict[cell][tf], 1)]
        else:
            tf_vec += [(0, 0)]
    tf_vec.sort()
    tf_locs = np.array([i[0] for i in tf_vec])
    tf_mask = np.array([i[1] for i in tf_vec])

    num_samples = input_.shape[2]

    # Pre-build these
    mask_feature = iio.make_int64_feature(tf_mask)
    name_feature = iio.make_bytes_feature([bytes(cell, 'utf-8')])

    for i in range(start, min(stop, num_samples)):

        x1 = input_[0:1000,:,i]
        x2 = np.array([[target[dnase_dict[cell], i]]] * 1000)
        x = np.append(x1, x2, 1)

        y = target[tf_locs, i] * tf_mask

        features = {}
        features['x/data'] = iio.make_int64_feature(x.flatten())
        features['x/shape'] = iio.make_int64_feature(x.shape)
        features['y'] = iio.make_int64_feature(y)
        features['mask'] = mask_feature
        features['cell'] = name_feature
        yield iio.make_example(features)
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default=0)
    parser.add_argument('--stop', default=1000)
    parser.add_argument('--cell', default='HeLa-S3')
    args = parser.parse_args()

    tmp = h5py.File('../../deepsea_train/train.mat')
    i, t = tmp['trainxdata'], tmp['traindata']
    iterator = records_iterator(i, t, args.cell, int(args.start), int(args.stop))

    compression_options = tf.python_io.TFRecordOptions(
        tf.python_io.TFRecordCompressionType.GZIP)
    if not glob.glob('./output/'+args.cell): 
        os.mkdir('./output/'+args.cell)

    iio.write_tfrecord(
        protos=iterator,
        path=os.path.join('./output/'+args.cell, args.start),
        options=compression_options)


if __name__ == '__main__':
    main()
