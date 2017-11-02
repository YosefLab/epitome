import numpy as np
import tensorflow as tf
import gym
import scipy.signal
import os
import time
import inspect
from multiprocessing import Process
import tensorflow.contrib.distributions as ds
from sklearn.utils import shuffle
from math import pi


################################# Define configuration ################################

LEARNING_RATE = ??
# number of glimpses per image
NUM_GLIMPSES = 6
# height, width to which glimpses get resized
GLIMPSE_SIZE = 12
# number of resolutions per glimpse
NUM_RESOLUTIONS = 3
# number of training epochs
NUM_EPOCHS = ??
# batch size for each training iterations
# total training iterations = num_epochs * number of images / batch_size
BATCH_SIZE =
# for normalization purposes
EPSILON = 1e-10
# dimensionality of hidden state vector
STATE_SIZE = 256
# dimensionality of location vector
LOC_SIZE = 2
# dimensionality of glimpse network output
# TODO better names for size of glimpse image/glimpse vector
GLIMPSE_VECTOR_SIZE = 256
# size of images
IMG_SIZE = 28
# number of classes for classification
NUM_CLASSES = 10
NUM_CHANNELS = 1
# standard deviation for Gaussian distribution of locations
STD_DEV = ??


################################# Define Networks ################################

def build_glimpse_network(
        data,
        location,
        scope,
        output_size=256,
        size=128,
        activation=tf.nn.relu,
        output_activation=tf.nn.relu):

    # do not want cross entropy gradient to flow through location network
    location = tf.stop_gradient(location)
    glimpses = get_glimpses(data, location)

    with tf.variable_scope(scope):
        h_l = tf.layers.dense(
            location,
            units=size,
            activation=activation,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.contrib.layers.xavier_initializer())
        h_g = tf.layers.dense(
            glimpses,
            units=size,
            activation=activation,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.contrib.layers.xavier_initializer())
        out_1 = tf.layers.dense(
            h_l,
            units=output_size,
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.contrib.layers.xavier_initializer())
        out_2 = tf.layers.dense(
            h_g,
            units=output_size,
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.contrib.layers.xavier_initializer())
        g_t = output_activation(out_1 + out_2)
    return g_t


def build_core_network(
        state,
        glimpse,
        scope,
        output_size=128,
        output_activation=tf.nn.relu):

    with tf.variable_scope(scope):
        out_1 = tf.layers.dense(
            state,
            units=output_size,
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.contrib.layers.xavier_initializer())
        out_2 = tf.layers.dense(
            glimpse,
            units=output_size,
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.contrib.layers.xavier_initializer())
        h_t = output_activation(out_1 + out_2)
    return h_t


def build_location_network(
        state,
        scope,
        output_size=2):

    # do not want RL gradient to flow through core, glimpse networks
    state = tf.stop_gradient(state)

    with tf.variable_scope(scope):
        mean = tf.layers.dense(
            state,
            units=output_size,
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.contrib.layers.xavier_initializer())
    return mean


def build_action_network(
        state,
        scope,
        output_size=10,
        output_activation=None):

    # output is not passed through softmax
    # tf.nn.softmax_cross_entropy_with_logits will do softmax
    with tf.variable_scope(scope):
        ac_probs = tf.layers.dense(
            state,
            units=output_size,
            activation=output_activation,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.contrib.layers.xavier_initializer())
    return ac_probs



def get_glimpses(data, loc):
    box_bounds = [
        [0.5, 0.5, 1, 1],
        [0.3, 0.3, 1, 1],
        [0.15, 0.15, 1, 1],
        [-.10, -.10, 1, 1]
    ]

    num_data_points = len(data)
    boxes = box_bounds * num_data_points
    box_ind = np.arange(num_data_points).repeat(NUM_RESOLUTIONS)
    crop_size = [GLIMPSE_SIZE, GLIMPSE_SIZE]
    method = "bilinear"
    extrapolation_value = 0.0

    data = np.expand_dims(data.reshape([num_data_points, IMG_SIZE, IMG_SIZE]), 3)

    glimpses = tf.image.crop_and_resize(
        data,
        boxes=boxes,
        box_ind=box_ind,
        crop_size=crop_size,
        method=method,
        extrapolation_value=extrapolation_value)

    # flatten each image into a vector
    # TODO concatenate all resolutions per image?
    return tf.reshape(tf.squeeze(glimpses), [num_data_points * NUM_RESOLUTIONS, GLIMPSE_SIZE * GLIMPSE_SIZE])


def get_location(location_output):
    # location network outputs the mean
    # TODO restrcit mean to be between (-1, 1)
    # sample from gaussian with above mean and predefined STD_DEV
    # TODO verify that this samples from multiple distributions
    return tf.distributions.Normal(loc=mean., scale=STD_DEV).sample(sample_shape=[LOC_SIZE, 1])


def get_action(action_output):
    # pass output through softmax
    softmax_output = tf.nn.softmax(action_output)
    # get action with highest probability
    return tf.argmax(softmax_output)


############################################ Train ############################################

def train():

    #========================================================================================#
    # Download Data
    #========================================================================================#
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    #========================================================================================#
    # Placeholders
    #========================================================================================#

    data_placeholder = tf.placeholder(shape=[None, IMG_SIZE * IMG_SIZE], name="data", dtype=tf.float32)
    labels_placeholder = tf.placeholder(shape=[None, NUM_CLASSES], name="labels", dtype=tf.int32)
    hidden_state_placeholder = tf.placeholder(shape=[None, STATE_SIZE], name="hidden_state", dtype=tf.float32)
    location_placeholder = tf.placeholder(shape=[None, LOC_SIZE], name="loc")

    #========================================================================================#
    # Tensorflow Engineering: Config, Session, Variable initialization
    #========================================================================================#

    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)

    sess = tf.Session(config=tf_config)
    sess.__enter__() # equivalent to `with sess:`
    tf.global_variables_initializer().run() #pylint: disable=E1101

    #========================================================================================#
    # Network Outputs
    #========================================================================================#

    g_t = build_glimpse_network(
        data=data_placeholder,
        location=location_placeholder,
        scope="glimpse")

    h_t = build_core_network(
        state=hidden_state_placeholder,
        glimpse=g_t,
        scope="core")

    action_output = build_action_network(
        state=h_t,
        scope="action")
    a_t = get_action(action_output)

    location_output = build_location_network(
        state=h_t,
        scope="location")
    l_t = sample(location_output)

    #========================================================================================#
    # Define Ops
    #========================================================================================#

    cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=a_t,
        logits=labels_placeholder)

    update_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy_loss)

    #========================================================================================#
    # Training Loop
    #========================================================================================#

    for i in range(NUM_EPOCHS):
        # TODO check if mnist.train.next_batch() has same functionality
        x_train, y_train = shuffle(mnist.train.images, mnist.train.labels, n_samples=BATCH_SIZE)
        for i in range(0,len(x_train), BATCH_SIZE):
            x_train_batch, y_train_batch = x_train[i:i+BATCH_SIZE]

            # initial state and location
            hidden_state = ??
            loc = ??
            # for each image in batch, take multiple glimpses
            for i in range(NUM_GLIMPSES):

                glimpses = get_glimpses(x_train_batch, loc)
                glimpse_vector = get_glimpse_vector(glimpses, loc)
