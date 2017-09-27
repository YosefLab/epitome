"""Library to read and write TFRecords."""

import tensorflow as tf
slim = tf.contrib.slim


def make_bytes_feature(value_list):
    """Creates a Tensorflow-Feature with bytes data.

    Args:
        value_list: A list of strings/bytestrings.

    Returns:
        A Tensorflow-Feature with bytes data.
    """
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=value_list))


def make_int64_feature(value_list):
    """Creates a Tensorflow-Feature with int data.

    Args:
        value_list: A list of ints.

    Returns:
        A Tensorflow-Feature with int data.
    """
    return tf.train.Feature(
        int64_list=tf.train.Int64List(value=value_list))


def make_float_feature(value_list):
    """Creates a Tensorflow-Feature with float data.

    Args:
        value_list: A list of strings/bytestrings.

    Returns:
        A Tensorflow-Feature with float data.
    """
    return tf.train.Feature(
        float_list=tf.train.FloatList(value=value_list))


def make_example(features_dict):
    """Creates a Tensorflow-Example.

    Args:
        feature_dict: A dict mapping keys to Tensorflow-Features.

    Returns:
        A Tensorflow-Example of the data.
    """
    return tf.train.Example(
        features=tf.train.Features(feature=features_dict))


def get_bytes_list(example, key):
    """Gets the bytes list from a key from an example

    Args:
        example: A Tensorflow-Example.
        key: str. A key that maps to a feature.

    Returns:
        A list of bytes in the feature the key maps to.
    """
    return example.features.feature[key].bytes_list.value


def get_int64_list(example, key):
    """Gets the int list from a key from an example

    Args:
        example: A Tensorflow-Example.
        key: str. A key that maps to a feature.

    Returns:
        A list of int in the feature the key maps to.
    """
    return example.features.feature[key].int64_list.value


def get_float_list(example, key):
    """Gets the float list from a key from an example

    Args:
        example: A Tensorflow-Example.
        key: str. A key that maps to a feature.

    Returns:
        A list of float in the feature the key maps to.
    """
    return example.features.feature[key].float_list.value


def read_tfrecord(path, proto=None, options=None):
    """Reads Tensorflow-Examples from a TFRecord.

    Args:
        path: The path to the TFRecord.
        proto: The protocol buffer being read in.
            If this is None, uses the Tensorflow-Example proto.
        options: A TFRecordOptions that specifies the compression
            scheme of the TFRecord being read. If this is None, it
            assumes no compression scheme is used.

    Yields:
        Tensorflow-Example protos from the TFRecord.
    """
    if proto is None:
        proto = tf.train.Example
    if options is None:
        options = tf.python_io.TFRecordOptions(
            tf.python_io.TFRecordCompressionType.NONE)
    string_iterator = tf.python_io.tf_record_iterator(path, options)
    for string in string_iterator:
        yield proto.FromString(string)


def write_tfrecord(protos, path, options=None):
    """Reads Tensorflow-Examples from a TFRecord.

    Args:
        proto: An iterable of protos.
        path: The path the write to.
        options: A TFRecordOptions that specifies the compression
            scheme used to write the TFRecord. If this is None, no
            compression scheme is used.
    """
    if options is None:
        options = tf.python_io.TFRecordOptions(
            tf.python_io.TFRecordCompressionType.NONE)
    writer = tf.python_io.TFRecordWriter(path, options)
    for proto in protos:
        writer.write(proto.SerializeToString())
    writer.close()
