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

LEARNING_RATE = 1e-1
# number of glimpses per image
NUM_GLIMPSES = 7
# height, width to which glimpses get resized
GLIMPSE_SIZE = 8
# number of resolutions per glimpse
NUM_RESOLUTIONS = 4
# number of training epochs
NUM_EPOCHS = 1000000
# batch size for each training iterations
# total training iterations = num_epochs * number of images / batch_size
BATCH_SIZE = 10
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
STD_DEV = 0.01


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
        output_size=STATE_SIZE,
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
        output_size=LOC_SIZE):

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


def get_boxes(loc):
    offset = GLIMPSE_SIZE / (LOC_SIZE * IMG_SIZE)
    scale = np.expand_dims(np.arange(1, NUM_RESOLUTIONS + 1), -1)
    first_corners = np.repeat(np.array([[-offset, -offset, offset, offset]]), NUM_RESOLUTIONS, 
        axis=0)
    all_corners = np.tile(first_corners * scale, reps=[BATCH_SIZE, 1])
    # repeated_locs = np.tolist(loc) * NUM_RESOLUTIONS
    repeated_loc = tf.reshape(tf.tile(loc, multiples=[1, NUM_RESOLUTIONS]), [NUM_RESOLUTIONS * BATCH_SIZE, LOC_SIZE])
    repeated_loc = tf.tile(repeated_loc, multiples=[1, 2])
    return all_corners + repeated_loc
    

def get_glimpses(data, loc):

    boxes = get_boxes(loc)
    box_ind = np.arange(BATCH_SIZE).repeat(NUM_RESOLUTIONS)
    crop_size = [GLIMPSE_SIZE, GLIMPSE_SIZE]
    method = "bilinear"
    extrapolation_value = 0.0

    data = tf.expand_dims(tf.reshape(data, [BATCH_SIZE, IMG_SIZE, IMG_SIZE]), 3)

    glimpses = tf.image.crop_and_resize(
        data,
        boxes=boxes,
        box_ind=box_ind,
        crop_size=crop_size,
        method=method,
        extrapolation_value=extrapolation_value)

    # flatten each image into a vector
    # TODO concatenate all resolutions per image?
    return tf.reshape(tf.squeeze(glimpses), [BATCH_SIZE, NUM_RESOLUTIONS * GLIMPSE_SIZE * GLIMPSE_SIZE])


def get_location(location_output):
    # location network outputs the mean
    # TODO restrcit mean to be between (-1, 1)
    # sample from gaussian with above mean and predefined STD_DEV
    # TODO verify that this samples from multiple distributions
    dist = tf.distributions.Normal(loc=location_output, scale=STD_DEV)
    samples = dist.sample(sample_shape=[1])
    return tf.squeeze(samples), tf.squeeze(dist.log_prob(samples))


def get_action(action_output):
    # pass output through softmax
    softmax_output = tf.nn.softmax(action_output)
    # get action with highest probability
    return tf.argmax(softmax_output, output_type=tf.int32)



if __name__ == '__main__':

    #========================================================================================#
    # Download Data
    #========================================================================================#
    
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    #========================================================================================#
    # Placeholders
    #========================================================================================#

    sy_x = tf.placeholder(shape=[None, IMG_SIZE * IMG_SIZE], 
        name="data", 
        dtype=tf.float32)

    sy_y = tf.placeholder(shape=[None, NUM_CLASSES], 
        name="labels", 
        dtype=tf.int32)

    sy_h = tf.placeholder(shape=[None, STATE_SIZE], 
        name="hidden_state", 
        dtype=tf.float32)

    sy_l = tf.placeholder(shape=[None, LOC_SIZE], 
        name="loc",
        dtype=tf.float32)

    #========================================================================================#
    # Hidden Cell
    #========================================================================================#

    glimpse_output = build_glimpse_network(
        data=sy_x,
        location=sy_l,
        scope="glimpse")

    hidden_output = build_core_network(
        state=sy_h,
        glimpse=glimpse_output,
        scope="core")

    raw_action_output = build_action_network(
        state=hidden_output,
        scope="action")
    action_output = get_action(raw_action_output)

    raw_location_output = build_location_network(
        state=hidden_output,
        scope="location")
    location_output, log_probs = get_location(raw_location_output)

    #========================================================================================#
    # Define Ops
    #========================================================================================#

    # cross entropy loss for actions that are output at final timestep 
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=sy_y,
        logits=raw_action_output))

    # learn weights for glimpse, core, and action network
    update_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy_loss)

    rewards = 1 - (1 * (action_output == tf.argmax(sy_y, output_type=tf.int32)))
    policy_gradient_loss = tf.scalar_mul(-1, tf.reduce_mean(log_probs * tf.to_float(rewards)))
    update_op_2 = tf.train.AdamOptimizer(LEARNING_RATE).minimize(policy_gradient_loss)

    #========================================================================================#
    # Tensorflow Engineering: Config, Session, Variable initialization
    #========================================================================================#
    
    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)

    sess = tf.Session(config=tf_config)
    sess.__enter__() # equivalent to `with sess:`
    tf.global_variables_initializer().run() #pylint: disable=E1101

    #========================================================================================#
    # Train
    #========================================================================================#

    location = np.random.uniform(size=[BATCH_SIZE, LOC_SIZE], low=-1.0, high=1.0)
    state = np.zeros(shape=[BATCH_SIZE, STATE_SIZE])
    
    for i in range(NUM_EPOCHS):
        
        # TODO check if mnist.train.next_batch() has same functionality
        x_train, y_train = shuffle(mnist.train.images, mnist.train.labels, n_samples=BATCH_SIZE)
        for i in range(0,len(x_train), BATCH_SIZE):
            x_train_batch, y_train_batch = x_train[i:i+BATCH_SIZE], y_train[i:i+BATCH_SIZE]

            for i in range(NUM_GLIMPSES):
                # not actually training 
                fetches = [action_output, raw_location_output, location_output, log_probs,
                        action_output, hidden_output, action_output, raw_action_output]
                outputs = sess.run(fetches=fetches, feed_dict={sy_x: x_train_batch, 
                    sy_y: y_train_batch, 
                    sy_l: location, 
                    sy_h: state})
                location = outputs[2]
                state = outputs[5]

                # print(outputs[-1])

    #========================================================================================#
    # Test
    #========================================================================================#
    
    # test()


