import numpy as np
import tensorflow as tf
import threading


"""
Lightweight library to read and write TF-Records.
"""

################################# TF-Record Utilities ################################

def make_bytes_feature(value_list):
  return tf.train.Feature(
    bytes_list=tf.train.BytesList(value=value_list))

def make_int64_feature(value_list):
  return tf.train.Feature(
    int64_list=tf.train.Int64List(value=value_list))

def make_float_feature(value_list):
  return tf.train.Feature(
    float_list=tf.train.FloatList(value=value_list))

def make_example(features_dict):
  return tf.train.Example(
    features=tf.train.Features(feature=features_dict))

def get_bytes_list(example, key):
  return example.features.feature[key].bytes_list.value

def get_int64_list(example, key):
  return example.features.feature[key].int64_list.value

def get_float_list(example, key):
  return example.features.feature[key].float_list.value

def read_tfrecord(path, proto=None, options=None):
  if proto is None:
    proto = tf.train.Example
  string_iterator = tf.python_io.tf_record_iterator(path, options)
  for string in string_iterator:
    yield proto.FromString(string)

def write_tfrecord(protos, path, options=None):
  writer = tf.python_io.TFRecordWriter(path, options)
  for proto in protos:
    writer.write(proto.SerializeToString())
  writer.close()


def write_tfrecords(examples, filename, num_shards=100, options = None):

    filenames = sharded_filenames(filename, num_shards)
    func = lambda file: tf.python_io.TFRecordWriter(file, options)
    writers = list(map(func, filenames))
    
    for i, example in enumerate(examples):
        t = threading.Thread(
            target=writers[i % num_shards].write,
            args=(example.SerializeToString(),))
        t.daemon = True
        t.start()
        t.join()


  ################################# Sharded Filespec ################################

def sharded_filename(filename, shard, num_shards):
  format_ = lambda i: str(i).rjust(5, '0')
  return '%s-%s-%s' % (filename, format_(shard), format_(num_shards))

def sharded_filenames(filename, num_shards):
  return [sharded_filename(filename, i, num_shards) for i in range(num_shards)]