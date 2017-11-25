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
import tensorflow.contrib.distributions as ds
from sklearn.utils import shuffle
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
        length,
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
        loc_size=loc_size,
        length=length)

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
        # clip mean to be in between -1 and 1
        return tf.clip_by_value(mean, -1, 1)


# TODO is this the best structure for baseline network?
def build_baseline_network(
        input_placeholder, 
        output_size,
        scope, 
        n_layers=2, 
        size=64, 
        activation=tf.tanh,
        output_activation=None):

    with tf.variable_scope(scope):
        out = input_placeholder
        # network has n_layers hidden layers
        for _ in range(n_layers):
            out = tf.layers.dense(
                out, 
                units=size, 
                activation=activation)
        return tf.layers.dense(
            out, 
            units=output_size, 
            activation=output_activation)


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
    

def get_glimpses(data, location, batch_size, num_resolutions, glimpse_size, img_size, loc_size,
    length):
    # first four channels of data are the one-hot encoded dna sequence
    dna = data[:, :, :4]

    # glimpse centered at start_index
    start_index = tf.to_int32(location)
    boolean_mask = get_boolean_mask(glimpse_size, start_index, length, batch_size)

    # pad DNA with -1 values on each side
    # new length of padded data is (2 * glimpse) + length
    padded_dna = get_padded_dna(dna, glimpse_size)

    # get mask into correct shape, tf.stack does weird things
    dna_boolean_mask = tf.squeeze(tf.stack([boolean_mask]*4, axis=-1), axis=2)
    sliced_dna = tf.boolean_mask(tensor=padded_dna, mask=dna_boolean_mask)
    sliced_dna = tf.reshape(sliced_dna, [batch_size, glimpse_size * 2, 4])

    # flatten sliced dna to get glimpse
    return tf.contrib.layers.flatten(sliced_dna)

        
def get_boolean_mask(glimpse_size, start_index, length, batch_size):
    curr_index = start_index
    padded_size = length + (2 * glimpse_size)
    index_mask = tf.one_hot(indices=curr_index, depth=padded_size, axis=1)
    for i in range((glimpse_size * 2) - 1):
        curr_index += 1
        index_mask += tf.one_hot(indices=curr_index, depth=padded_size, axis=1)
    return index_mask > 0
    

def get_padded_glimspe(glimpse, glimpse_size):
    return tf.pad(glimpse, paddings=[[0, 0], [glimpse_size, glimpse_size], [0, 0]], 
        constant_values=-1)


def get_padded_dna(dna, glimpse_size):
    return get_padded_glimspe(dna, glimpse_size)


def get_location(location_output, std_dev, loc_size, length, clip=False, clip_low=-1, clip_high=1):
    # location network outputs the mean
    # TODO restrcit mean to be between (-1, 1)
    # sample from gaussian with above mean and predefined STD_DEV
    # TODO verify that this samples from multiple distributions
    dist = ds.MultivariateNormalDiag(loc=location_output, scale_diag=[std_dev] * loc_size)
    samples = tf.squeeze(dist.sample(sample_shape=[1]), axis=0)
    if clip:
        samples = tf.clip_by_value(samples, clip_low, clip_high)
    # locations for sequences should be between 0 and length
    samples = (samples + 1) * (length/2)
    return samples, tf.squeeze(dist.log_prob(tf.expand_dims(samples, -1)))


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
        nn_baseline,
        num_epochs,
        learning_rate,
        batch_size,
        loc_size, 
        img_size, 
        num_classes,
        num_channels):

    ################################# Download data #############################
    
    # lenght of the region in the genome
    length = 1000
    num_tfs = 919

    # TODO do not hard code this
    num_classes = num_tfs

    # deepsea sequence-only data
    import deepsea_data

    ################################# Placeholders ##############################

    sy_x = tf.placeholder(shape=[None, length, 4], 
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
        length=length,
        scope="glimpse")

    hidden_output = build_core_network(
        state=sy_h,
        glimpse=glimpse_output,
        scope="core",
        output_size=state_size)

    baseline_output = build_baseline_network(
        input_placeholder=hidden_output, 
        output_size=1,
        scope="baseline", 
        n_layers=2, 
        size=64, 
        activation=tf.tanh,
        output_activation=None)

    raw_action_output = build_action_network(
        state=hidden_output,
        scope="action",
        output_size=num_classes)
    d, action_output = get_action(raw_action_output)

    mean_location_output = build_location_network(
        state=hidden_output,
        scope="location",
        output_size=loc_size)
    location_output, log_probs = get_location(mean_location_output, std_dev, loc_size, length,
        clip=True)

    ################################# Define ops ################################

    # cross entropy loss for actions that are output at final timestep 

    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=sy_y,
        logits=raw_action_output))

    rewards = tf.cast(tf.equal(action_output, tf.argmax(sy_y, output_type=tf.int32, axis = 1)), tf.float32)

    # if use baseline, subtract baseline from rewards
    if nn_baseline:
        baseline_prediction = tf.squeeze(baseline_output)
        adv_n = rewards - tf.stop_gradient(baseline_prediction)
    else:
        adv_n = rewards

    policy_gradient_loss = tf.scalar_mul(-1, tf.reduce_mean(log_probs * tf.to_float(adv_n)))

    if nn_baseline:
        baseline_loss = tf.losses.mean_squared_error(rewards, baseline_prediction)
        hybrid_loss = cross_entropy_loss + policy_gradient_loss + baseline_loss
    else:
        hybrid_loss = cross_entropy_loss + policy_gradient_loss

    update_op = tf.train.AdamOptimizer(learning_rate).minimize(hybrid_loss)

    ############################## Tensorflow engineering #######################
    
    # initialize config 
    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)

    # initialize session and variables
    sess = tf.Session(config=tf_config)
    sess.__enter__() # equivalent to `with sess:`
    tf.global_variables_initializer().run() # pylint: disable=E1101

    ################################### Train ###################################

    # hidden state initialized to zeros
    state = np.zeros(shape=[batch_size, state_size])
    location = np.zeros(shape=[batch_size, loc_size])
    
    for epoch in range(num_epochs):
        
        # accuracies
        acs = []
        # cross entropy and policy gradient losses
        ce_losses = []
        pg_losses = []
        baseline_losses = []
        # total rewards for num_glimpses timesteps
        path_rewards = []

        # TODO do not hardcode this file path
        train_batches = deepsea_data.train_iterator(
            source='../../deepsea_train/train.mat',
            batch_size=batch_size,
            num_epochs=1)

        for x_train_batch, y_train_batch in train_batches:
            print(x_train_batch.shape, y_train_batch.shape, state.shape)
            for j in range(num_glimpses - 1):        
                fetches = [location_output, hidden_output]              
                outputs = sess.run(fetches=fetches, feed_dict={
                    sy_x: x_train_batch, 
                    sy_y: y_train_batch, 
                    sy_l: location, 
                    sy_h: state})
                location = outputs[0]
                state = outputs[1]
            
            fetches = [location_output, hidden_output, update_op, cross_entropy_loss,
                policy_gradient_loss, rewards]
            
            if nn_baseline:
                fetches.append(baseline_loss)
            # make sure action_output is last in outputs
            fetches.append(action_output)
            outputs = sess.run(fetches=fetches, feed_dict={sy_x: x_train_batch, 
                sy_y: y_train_batch, 
                sy_l: location, 
                sy_h: state})
            correct_prediction = np.mean(np.equal(np.argmax(y_train_batch, axis=1), outputs[-1]))
            acs.append(correct_prediction)
            ce_losses.append(outputs[3])
            pg_losses.append(outputs[4])
            path_rewards.append(outputs[5])
            if nn_baseline:
                baseline_losses.append(outputs[6])

        ######################### Print out epoch stats ########################
        
        print("*" * 100)
        print("Epoch: {}".format(epoch))
        print("Accuracy: {}".format(np.mean(np.array(acs))))
        print("Cross Entropy Loss: {}".format(np.mean(np.array(ce_losses))))
        print("Policy Gradient Loss: {}".format(np.mean(np.array(pg_losses))))
        if nn_baseline:
            print("Baseline Loss: {}".format(np.mean(np.array(baseline_losses))))
        print("Rewards: {}".format(np.mean(np.array(path_rewards))))


# constant for normalization purposes
EPSILON = 1e-10


def main():
    parser = argparse.ArgumentParser()
    
    ########################## Model architecture args ##########################

    # height, width to which glimpses get resized
    parser.add_argument('--glimpse_size', type=int, default=8)
    # number of glimpses per image
    parser.add_argument('--num_glimpses', type=int, default=4)
    # number of resolutions per glimpse
    parser.add_argument('--num_resolutions', type=int, default=4) 
    # dimensionality of glimpse network output
    # TODO better names for size of glimpse image/glimpse vector
    parser.add_argument('--glimpse_vector_size', type=int, default=256)
    # dimensionality of hidden state vector
    parser.add_argument('--state_size', type=int, default=256)
    # standard deviation for Gaussian distribution over locations
    parser.add_argument('--std_dev', '-std', type=int, default=1e-3)
    # use neural network baseline
    parser.add_argument('--nn_baseline', '-bl', action='store_true')

    ############################## Training args ################################

    # number of full passes through the data
    # total training iterations = num_epochs * number of images / batch_size
    parser.add_argument('--num_epochs', type=int, default=100000)
    parser.add_argument('--learning_rate', '-lr', type=int, default=1e-2)
    # batch size for each training iterations
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    # random seed for deterministic training
    parser.add_argument('--random_seed', '-rs', type=int, default=42)

    ############################## Input data args ##############################

    # Defaults for these arguments are set for MNIST
    # Will need to update for DNA sequence inputs

    # dimensionality of location vector
    parser.add_argument('--loc_size', type=int, default=1)
    # original size of images
    parser.add_argument('--img_size', type=int, default=28)
    # number of classes for classification
    parser.add_argument('--num_classes', type=int, default=2)
    # number of channels in the input data
    parser.add_argument('--num_channels', type=int, default=1)
    args = parser.parse_args()
    
    # setting random seed
    np.random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)

    train(glimpse_size=args.glimpse_size, 
        num_glimpses=args.num_glimpses,
        num_resolutions=args.num_resolutions,
        glimpse_vector_size=args.glimpse_vector_size,
        state_size=args.state_size,
        std_dev=args.std_dev,
        nn_baseline = args.nn_baseline,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        loc_size=args.loc_size, 
        img_size=args.img_size, 
        num_classes=args.num_classes,
        num_channels=args.num_channels)


if __name__ == "__main__":
    main()
