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
import os
import time
import tensorflow.contrib.distributions as ds
from sklearn.utils import shuffle
from math import pi
import tensorflow as tf
import argparse


################################# Define Networks ################################

def build_glimpse_network(
        data,
        location,
        batch_size, 
        num_resolutions, 
        glimpse_size, 
        img_size,
        loc_size,
        scope,
        output_size=256,
        size=128,
        activation=tf.nn.relu,
        output_activation=tf.nn.relu):

    # do not want cross entropy gradient to flow through location network
    location = tf.stop_gradient(location)
    glimpses = get_glimpses(data=data, 
        location=location, 
        batch_size=batch_size, 
        num_resolutions=num_resolutions, 
        glimpse_size=glimpse_size, 
        img_size=img_size,
        loc_size=loc_size)

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
        output_size,
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
        output_size):

    # do not want RL gradient to flow through core, glimpse networks
    state = tf.stop_gradient(state)

    with tf.variable_scope(scope):
        mean = tf.layers.dense(
            state,
            units=output_size,
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.contrib.layers.xavier_initializer())
    return tf.clip_by_value(mean, -1, 1)


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
    

def get_glimpses(data, location, batch_size, num_resolutions, glimpse_size, img_size, loc_size):

    boxes = get_boxes(location, batch_size, num_resolutions, glimpse_size, img_size, loc_size)
    box_ind = np.arange(batch_size).repeat(num_resolutions)
    crop_size = [glimpse_size, glimpse_size]
    method = "bilinear"
    extrapolation_value = 0.0

    data = tf.expand_dims(tf.reshape(data, [batch_size, img_size, img_size]), 3)

    glimpses = tf.image.crop_and_resize(
        data,
        boxes=boxes,
        box_ind=box_ind,
        crop_size=crop_size,
        method=method,
        extrapolation_value=extrapolation_value)

    # flatten each image into a vector
    # TODO concatenate all resolutions per image?
    return tf.reshape(tf.squeeze(glimpses), 
        [batch_size, num_resolutions * glimpse_size * glimpse_size])


def get_boxes(loc, batch_size, num_resolutions, glimpse_size, img_size, loc_size):
    offset = glimpse_size / (loc_size * img_size)
    scale = np.expand_dims(np.arange(1, num_resolutions + 1), -1)
    first_corners = np.repeat(np.array([[-offset, -offset, offset, offset]]), num_resolutions, 
        axis=0)
    all_corners = np.tile(first_corners * scale, reps=[batch_size, 1])
    # repeated_locs = np.tolist(loc) * NUM_RESOLUTIONS
    repeated_loc = tf.reshape(tf.tile(loc, multiples=[1, num_resolutions]),
        [num_resolutions * batch_size, loc_size])
    repeated_loc = tf.tile(repeated_loc, multiples=[1, 2])
    return all_corners + repeated_loc


def get_location(location_output, std_dev, clip=False, clip_low=-1, clip_high=1):
    # location network outputs the mean
    # TODO restrcit mean to be between (-1, 1)
    # sample from gaussian with above mean and predefined STD_DEV
    # TODO verify that this samples from multiple distributions
    dist = tf.distributions.Normal(loc=location_output, scale=std_dev)
    samples = tf.squeeze(dist.sample(sample_shape=[1]))
    if clip:
        samples = tf.clip_by_value(samples, clip_low, clip_high)
    return samples, tf.squeeze(dist.log_prob(samples))


def get_action(action_output):
    # pass output through softmax
    softmax_output = tf.nn.softmax(action_output)
    # get action with highest probability
    return softmax_output, tf.argmax(softmax_output, output_type=tf.int32, axis=-1)


def train(glimpse_size, 
        num_glimpses,
        num_resolutions,
        glimpse_vector_size,
        state_size,
        std_dev,
        num_epochs,
        learning_rate,
        batch_size,
        loc_size, 
        img_size, 
        num_classes,
        num_channels):

    ################################# Download data #############################
    
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    ################################# Placeholders ##############################

    sy_x = tf.placeholder(shape=[None, img_size * img_size], 
        name="data", 
        dtype=tf.float32)

    sy_y = tf.placeholder(shape=[None, num_classes], 
        name="labels", 
        dtype=tf.int32)

    sy_h = tf.placeholder(shape=[None, state_size], 
        name="hidden_state", 
        dtype=tf.float32)

    sy_l = tf.placeholder(shape=[None, loc_size], 
        name="loc",
        dtype=tf.float32)

    ################################# RNN cell ##################################

    glimpse_output = build_glimpse_network(
        data=sy_x,
        location=sy_l,
        batch_size=batch_size, 
        num_resolutions=num_resolutions, 
        glimpse_size=glimpse_size, 
        img_size=img_size,
        loc_size=loc_size,
        scope="glimpse")

    hidden_output = build_core_network(
        state=sy_h,
        glimpse=glimpse_output,
        scope="core",
        output_size=state_size)

    raw_action_output = build_action_network(
        state=hidden_output,
        scope="action")
    d, action_output = get_action(raw_action_output)

    raw_location_output = build_location_network(
        state=hidden_output,
        scope="location",
        output_size=loc_size)
    location_output, log_probs = get_location(raw_location_output, std_dev, clip=True)

    ################################# Define ops ################################

    # cross entropy loss for actions that are output at final timestep 
    # cross_entropy_loss = tf.reduce_mean(tf.reduce_max(tf.log(d), axis=1))

    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=sy_y,
        logits=raw_action_output))

    # learn weights for glimpse, core, and action network
    # update_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy_loss)

    # rewards = 1 * tf.equal(action_output , tf.argmax(sy_y, output_type=tf.int32))
    # policy_gradient_loss = tf.scalar_mul(-1, tf.reduce_mean(log_probs * tf.to_float(rewards)))
    # tf.equal(action_output, )
    update_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)

    ############################## Tensorflow engineering #######################
    
    # initialize config 
    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)

    # initialize session and variables
    sess = tf.Session(config=tf_config)
    sess.__enter__() # equivalent to `with sess:`
    tf.global_variables_initializer().run() # pylint: disable=E1101

    ################################### Train ###################################

    location = np.random.uniform(size=[batch_size, loc_size], low=-0.5, high=0.5)
    # location = np.zeros([batch_size, loc_size])
    state = np.zeros(shape=[batch_size, state_size])
    
    for epoch in range(num_epochs):
        
        acs = []
        losses = []
        
        x_train, y_train = shuffle(mnist.train.images, mnist.train.labels)
        for i in range(0, len(x_train), batch_size):
            x_train_batch, y_train_batch = x_train[i:i+batch_size], y_train[i:i+batch_size]


            for j in range(num_glimpses - 1):
                # not actually training 
                fetches = [location_output, hidden_output, cross_entropy_loss, raw_location_output, raw_action_output]
                outputs = sess.run(fetches=fetches, feed_dict={sy_x: x_train_batch, 
                    sy_y: y_train_batch, 
                    sy_l: location, 
                    sy_h: state})
                location = np.random.uniform(size=[batch_size, loc_size], low=-0.5, high=0.5)
                # location = outputs[0]
                state = outputs[1]

            fetches = [location_output, hidden_output, update_op, cross_entropy_loss, action_output]
            outputs = sess.run(fetches=fetches, feed_dict={sy_x: x_train_batch, 
                    sy_y: y_train_batch, 
                    sy_l: location, 
                    sy_h: state})

            correct_prediction = np.sum(np.equal(np.argmax(y_train_batch, axis=1), outputs[-1]))/batch_size
            acs.append(correct_prediction)
            losses.append(outputs[3])
        
        print("*" * 100)
        print("Epoch: {}".format(epoch))
        print("Accuracy: {}".format(np.mean(np.array(acs))))
        print("Cross Entropy Loss: {}".format(np.mean(np.array(losses))))


# constant for normalization purposes
EPSILON = 1e-10


def main():
    parser = argparse.ArgumentParser()
    
    ########################## Model architecture args ##########################

    # height, width to which glimpses get resized
    parser.add_argument('--glimpse_size', type=int, default=8)
    # number of glimpses per image
    parser.add_argument('--num_glimpses', type=int, default=7)
    # number of resolutions per glimpse
    parser.add_argument('--num_resolutions', type=int, default=4)
    # dimensionality of hidden state vector
    # dimensionality of glimpse network output
    # TODO better names for size of glimpse image/glimpse vector
    parser.add_argument('--glimpse_vector_size', type=int, default=256)
    # dimensionality of hidden state vector
    parser.add_argument('--state_size', type=int, default=256)
    # standard deviation for Gaussian distribution over locations
    parser.add_argument('--std_dev', '-std', type=int, default=1e-3)

    ############################## Training args ################################

    # number of full passes through the data
    # total training iterations = num_epochs * number of images / batch_size
    parser.add_argument('--num_epochs', type=int, default=100000)
    parser.add_argument('--learning_rate', '-lr', type=int, default=1e-2)
    # batch size for each training iterations
    parser.add_argument('--batch_size', '-b', type=int, default=1000)

    ############################## Input data args ##############################

    # Defaults for these arguments are set for MNIST
    # Will need to update for DNA sequence inputs

    # dimensionality of location vector
    parser.add_argument('--loc_size', type=int, default=2)
    # original size of images
    parser.add_argument('--img_size', type=int, default=28)
    # number of classes for classification
    parser.add_argument('--num_classes', type=int, default=10)
    # number of channels in the input data
    parser.add_argument('--num_channels', type=int, default=1)
    args = parser.parse_args()
    
    train(glimpse_size=args.glimpse_size, 
        num_glimpses=args.num_glimpses,
        num_resolutions=args.num_resolutions,
        glimpse_vector_size=args.glimpse_vector_size,
        state_size=args.state_size,
        std_dev=args.std_dev,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        loc_size=args.loc_size, 
        img_size=args.img_size, 
        num_classes=args.num_classes,
        num_channels=args.num_channels)


if __name__ == "__main__":
    main()

