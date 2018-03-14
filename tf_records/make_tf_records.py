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

# TODO: get allpos_bed file, use this to get regions in accessibility_path
def records_iterator(input_, target, cell, start, stop, features_path, accessibility_path
    ):

    # Builds dicts of indicies of different features
    dnase_dict, tf_dict = parse_feature_name(features_path)
    
    # builds the vector of locations for querying from matrix, and the mask
    tf_vec = []
    i = 0
    for tf in tfs:
        if tf in tf_dict[cell]:
            tf_vec += [(tf_dict[cell][tf], 1)]
        else:
            tf_vec += [(0, 0)]
    tf_locs = np.array([v[0] for v in tf_vec])
    tf_mask = np.array([v[1] for v in tf_vec])

    # Pre-build these features
    mask_feature = iio.make_int64_feature(tf_mask)
    name_feature = iio.make_bytes_feature([bytes(cell, 'utf-8')])

    num_samples = input_.shape[2]
    for i in range(start, min(stop, num_samples)):

        # x is 1000 * 5, where the last row is all 0s or 1s depending on the
        # dnase label
        x1 = input_[0:1000,:,i]
        x2 = np.array([[target[dnase_dict[cell], i]]] * 1000)
        x = np.append(x1, x2, 1)

        # y is queried from the target matrix. We mask by whether we actually
        # have this TF for this cell type.
        y = np.array([target[c, i] for c in tf_locs]) * tf_mask

        # The features going into the example.
        features = {}
        # x has to be flattened.
        features['x/data'] = iio.make_int64_feature(x.flatten())
        features['x/shape'] = iio.make_int64_feature(x.shape)
        features['y'] = iio.make_int64_feature(y)
        # We'll use mask in the loss function. 
        features['mask'] = mask_feature
        features['cell'] = name_feature

        yield iio.make_example(features)
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default=0)
    parser.add_argument('--stop', default=1000)
    parser.add_argument('--cell', default='HeLa-S3')
    # On millennium, these paths should not be in the home directory!!!!
    parser.add_argument('--features', 
        default='../../DeepSEA-v0.94/resources/feature_name')
    parser.add_argument('--accessibility', 
        default='/data/epitome/accessibility/dnase/hg19')
    parser.add_argument('--out', default='./output')
    parser.add_argument('--data', default='../../deepsea_train/train.mat')
    args = parser.parse_args()


    output_dir = os.path.join(args.out, args.cell)
    if not glob.glob(args.out):
        os.mkdir(args.out)
    if not glob.glob(output_dir):
        os.mkdir(output_dir)


    tmp = h5py.File(args.data)
    i, t = tmp['trainxdata'], tmp['traindata']
    
    

    # traverse through all files in accessibility directory and get filenames
    accessibility_files = [f for f in os.listdir(args.accessibility) if os.isfile(os.join(args.accessibility, f))]
    # get accessibility file with correct cell type
    accessibility_path = filter(lambda x: x.contains(args.cell, accessibility_files)
                                
    # there should be accessibility
    assert(len(accessibility_path) == 1)
                    

    iterator = records_iterator(i, t, args.cell, int(args.start),
     int(args.stop), features_path=args.features, accessibility = accessibility_path[0])
    compression_options = tf.python_io.TFRecordOptions(
        tf.python_io.TFRecordCompressionType.GZIP)

    iio.write_tfrecord(
        protos=iterator,
        path=os.path.join(output_dir, args.start.zfill(7)),
        options=compression_options)


if __name__ == '__main__':
    main()
