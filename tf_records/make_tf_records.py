### Creates TF records from DEEPSEA data ###

import epitome.iio as iio
from epitome.functions import *
from epitome.constants import *
from epitome.generators import *

import argparse
import threading
import random
import glob

def main():
    """
    Writes tf records for dataset. 
    """
    
    parser = argparse.ArgumentParser()
    # array of cell types to be tested
    parser.add_argument('--test_celltypes', nargs='+', help='test celltypes', default="A549")
    parser.add_argument('--sample', help='rate to sample records by', default=0.1, type=float)
    parser.add_argument('--seed', help='random seet to sample records', default=3)
    
    # On millennium, these paths should not be in the home directory!!!!
    parser.add_argument('--feature_path', 
        default='../../DeepSEA-v0.94/resources/feature_name')
    parser.add_argument('--output_dir', default='./output')
    parser.add_argument('--data', default='../../deepsea_train')
    args = parser.parse_args()
    
    # create output dir if it does not exist
    if not glob.glob(args.output_dir):
        os.mkdir(args.output_dir)
        
    train_data, valid_data, test_data = load_deepsea_label_data(args.data)
    
    
    matrix, cellmap, assaymap = get_assays_from_feature_file(feature_path=args.feature_path, 
                                  eligible_assays = None,
                                  eligible_cells = None, min_cells_per_assay = 2, min_assays_per_cell=5)
    
    eval_cell_types = list(cellmap).copy()
    
    test_cell_types = args.test_celltypes

    [eval_cell_types.remove(test_cell) for test_cell in test_cell_types]
    print("eval cell types", eval_cell_types)
    print("test cell types", test_cell_types)
    
    # set sampling seed
    random.seed(args.seed)
            
    # make datasets for train, valid and test
    radii = [1,3,10,30]
    indices = random.sample(range(train_data["y"].shape[1]), int(train_data["y"].shape[1] * args.sample))
    train_iter = gen_from_peaks_to_tf_records(train_data,  
                                            eval_cell_types,
                                            eval_cell_types,
                                            matrix,
                                            assaymap,
                                            cellmap, 
                                            radii = radii, 
                                            mode = Dataset.TRAIN, indices = indices)()

    indices = random.sample(range(valid_data["y"].shape[1]), int(valid_data["y"].shape[1] * args.sample))
    valid_iter = gen_from_peaks_to_tf_records(valid_data, 
                                            eval_cell_types,
                                            eval_cell_types,
                                            matrix,
                                            assaymap,
                                            cellmap,
                                            radii = radii, 
                                            mode = Dataset.VALID, indices = indices)()

    # Don't sample test set
    test_iter = gen_from_peaks_to_tf_records(test_data, 
                                           test_cell_types, 
                                           eval_cell_types,
                                           matrix,
                                           assaymap,
                                           cellmap, 
                                           radii = radii, 
                                           mode = Dataset.TEST)()
    
    # set compression to gzipped
    compression_options = tf.python_io.TFRecordOptions(
        tf.python_io.TFRecordCompressionType.GZIP)
    
    # write train
    print("writing gzipped train records")
    iio.write_tfrecords(train_iter, filename=os.path.join(args.output_dir, 'train.tfrecord'),
        num_shards = 500, options=compression_options)
    
    # write validation
    print("writing gzipped validation records")
    iio.write_tfrecords(valid_iter, filename=os.path.join(args.output_dir, 'valid.tfrecord'),
        num_shards = 10, options=compression_options)
    
    # write test
    print("writing gzipped test records")       
    iio.write_tfrecords(test_iter, filename=os.path.join(args.output_dir, 'test.tfrecord'),
        num_shards = 50, options=compression_options)
    
if __name__ == '__main__':
    main()

