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
import tensorflow as tf
import argparse
import h5py


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
        dna_dim,
        deepsea,
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
        length=length,
        dna_dim=dna_dim,
        deepsea=deepsea)

    assert(glimpses.shape[1] == (glimpse_size * 2) * (dna_dim + num_resolutions))

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
        scope,
        output_size,
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
  
    # output is not passed through sigmoid
    # tf.nn.sigmoid_cross_entropy_with_logits uses raw outputs and will do sigmoid
    # we also pass through sigmoid later
    with tf.variable_scope(scope):
        ac_probs = tf.layers.dense(
            state,
            units=output_size,
            activation=output_activation,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.contrib.layers.xavier_initializer())
        return ac_probs


def get_glimpses(data, location, batch_size, num_resolutions, glimpse_size, img_size, loc_size,
    length, dna_dim, deepsea):
    # first four channels of data are the one-hot encoded dna sequence
    dna = data[:, :, :dna_dim]
    # rest of channels are the ATAC-seq data
    data = data[:, :, dna_dim:]
    
    # max pool ATAC-seq data at different resolutions
    glimpses = []
    for i in range(num_resolutions):
        resolution = 2**i
        glimpse = tf.nn.pool(
            input=data,
            window_shape=[resolution],
            strides=[resolution],
            pooling_type="MAX",
            padding="SAME")
        glimpses.append(glimpse)
    
    # combine DNA and ATAC data, slice to right size, return glimpses
    return index_glimpses(dna, location, num_resolutions, glimpses, glimpse_size, 
        length, batch_size, dna_dim, deepsea)


def concatenate_dna(boolean_mask, dna, glimpse_size, batch_size, dna_dim):
    padded_dna = get_padded_dna(dna, glimpse_size)
    # get mask into correct shape, tf.stack does weird things
    dna_boolean_mask = tf.squeeze(tf.stack([boolean_mask] * dna_dim, axis=-1), axis=2)
    sliced_dna = tf.boolean_mask(tensor=padded_dna, mask=dna_boolean_mask)
    sliced_dna = tf.reshape(sliced_dna, [batch_size, glimpse_size * 2, dna_dim])
    return sliced_dna


def index_glimpses(dna, location, num_resolutions, glimpses, glimpse_size, length, batch_size, dna_dim, deepsea):
    to_concatenate = []

    # we never enter loop with deepsea data because num_resolutions is zero
    for i in range(num_resolutions):
        glimpse = glimpses[i]
        # glimpse centered at start_index
        start_index = tf.to_int32(location / 2.0**i)
        boolean_mask = get_boolean_mask(glimpse_size, start_index, glimpse.shape[1], batch_size)
        
        # pad ATAC-seq and DNA with -1 values on each side
        # new length of padded data is (2 * glimpse) + length
        padded_glimpse = get_padded_glimspe(glimpse, glimpse_size)
        
        # concatenate DNA
        if i == 0:
            sliced_dna = concatenate_dna(boolean_mask, dna, glimpse_size, batch_size, dna_dim)
            to_concatenate.append(sliced_dna)

        # TODO look into why this line caused a bug
        sliced_glimpse = tf.boolean_mask(tensor=padded_glimpse, mask=boolean_mask)
        sliced_glimpse = tf.reshape(sliced_glimpse, [batch_size, glimpse_size * 2])
        sliced_glimpse = tf.expand_dims(sliced_glimpse, axis=-1)
        to_concatenate.append(sliced_glimpse)

    if deepsea:
        start_index = tf.to_int32(location)
        boolean_mask = get_boolean_mask(glimpse_size, start_index, length, batch_size)
        sliced_dna = concatenate_dna(boolean_mask, dna, glimpse_size, batch_size, dna_dim)
        to_concatenate.append(sliced_dna)
        assert(len(to_concatenate) == 1)

    # flatten all channels
    flat_shape = [batch_size, (num_resolutions + dna_dim) * (glimpse_size * 2)]
    return tf.reshape(tf.concat(to_concatenate, axis=-1), flat_shape)

        
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
        constant_values=0)


def get_padded_dna(dna, glimpse_size):
    return get_padded_glimspe(dna, glimpse_size)


def get_location(location_output, std_dev, loc_size, length, clip=False, clip_low=-1, clip_high=1):
    # location network outputs the mean
    # sample from gaussian with above mean and predefined STD_DEV
    dist = ds.MultivariateNormalDiag(loc=location_output, scale_diag=[std_dev] * loc_size)
    samples = tf.squeeze(dist.sample(sample_shape=[1]), axis=0)
    if clip:
        samples = tf.clip_by_value(samples, clip_low, clip_high)
    # locations for sequences should be between 0 and length - 1
    samples = (samples + 1) * (length/2)
    # TODO how can we avoid using tf.clip
    # clip again just to make sure indices are valid
    samples = tf.clip_by_value(samples, 0, length - 1)
    return samples, tf.expand_dims(dist.log_prob(samples), -1)


def get_action(raw_action_output, threshold):
    # pass output through activation
    # sigmoid for sequence data
    sigmoid_output = tf.nn.sigmoid(raw_action_output)
    return sigmoid_output, tf.cast(sigmoid_output > threshold, tf.int32)


####################################### Train ###################################


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
        num_channels,
        length,
        threshold,
        deepsea):

    ################################# Download data #############################

    if deepsea:
        import deepsea_data
        # force num_resolutions to be zero, no ATAC data
        num_resolutions = 0
        dna_dim = 4
        num_classes = 919
        # TODO do not hardcode this file path
        train_batches = deepsea_data.train_iterator(
            source='../../deepsea_train/train.mat',
            batch_size=batch_size,
            num_epochs=1)
        loc_size = 1
    else:
        # TODO remove file hardcoding later and use all files
        data = h5py.File("results-hdf5/CEBPB-A549.hdf5", "r")
        dna = data["seq"]
        atac = np.expand_dims(data["atac"], -1)
        # dna_dim is either four (ATCG) or five (ATCGN)
        dna_dim = dna.shape[-1]
        x_train = np.concatenate([dna, atac], axis=-1)
        y_train = data["label"]
        train_batches = range(0, len(x_train), batch_size)

        # TODO change num_classes to num_labels
        # y_train should be two dimensional (n x num_classes)
        assert(len(y_train.shape) == 2)
        # override default num_classes based on data
        num_classes = y_train.shape[1]
        loc_size = 1

    ################################# Placeholders ##############################

    # want additional channels to be 0 for deepsea data
    atac_channel = 1 if num_resolutions > 0 else 0
    sy_x = tf.placeholder(shape=[None, length, dna_dim + atac_channel],
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
        scope="glimpse",
        dna_dim=dna_dim,
        deepsea=deepsea,
        output_size=glimpse_vector_size)

    hidden_output = build_core_network(
        state=sy_h,
        glimpse=glimpse_output,
        scope="core",
        output_size=state_size)

    baseline_output = build_baseline_network(
        input_placeholder=hidden_output, 
        scope="baseline",
        output_size=1)

    raw_action_output = build_action_network(
        state=hidden_output,
        scope="action",
        output_size=num_classes)

    # sigmoid_output is raw network output passed through sigmoid activation
    # action_output is final classification based on threshold
    sigmoid_output, action_output = get_action(raw_action_output, threshold)

    mean_location_output = build_location_network(
        state=hidden_output,
        scope="location",
        output_size=loc_size)
    location_output, log_probs = get_location(mean_location_output, std_dev, loc_size, length, clip=True)

    ################################# Define ops ################################

    # cross entropy loss for actions that are output at final timestep
    cross_entropy_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.cast(sy_y, tf.float32),
        logits=raw_action_output))

    # reduce sum across different TFs
    # rewards should be number of correct predictions for each example
    rewards = tf.expand_dims(tf.reduce_sum(tf.cast(tf.equal(action_output, sy_y), tf.float32), axis=-1), axis=-1)

    # if use baseline, subtract baseline from rewards
    if nn_baseline:
        baseline_prediction = baseline_output
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
    sess.__enter__()  # equivalent to `with sess:`
    tf.global_variables_initializer().run()  # pylint: disable=E1101

    ################################### Train ###################################

    # hidden state initialized to zeros
    state = np.zeros(shape=[batch_size, state_size])
    # TODO what should initial location be?
    location = np.zeros(shape=[batch_size, loc_size]) + (length/2.0)
    
    for epoch in range(num_epochs):
        
        # accuracies
        acs = []
        # cross entropy and policy gradient losses
        ce_losses = []
        pg_losses = []
        baseline_losses = []
        # total rewards for num_glimpses timesteps
        path_rewards = []

        # train_batches defined above based on data type
        for i in train_batches:

            x_train_batch = i[0] if deepsea else x_train[i:i+batch_size]
            y_train_batch = i[1] if deepsea else y_train[i:i+batch_size]
            
            for j in range(num_glimpses - 1):        
                fetches = [location_output, hidden_output]
                outputs = sess.run(fetches=fetches, feed_dict={sy_x: x_train_batch,
                    sy_y: y_train_batch, 
                    sy_l: location, 
                    sy_h: state})

                location = outputs[0]
                state = outputs[1]

            fetches = [update_op, glimpse_output, location_output, mean_location_output, log_probs, hidden_output,
                       raw_action_output, sigmoid_output, rewards, cross_entropy_loss, policy_gradient_loss]

            if nn_baseline:
                fetches.append(baseline_loss)
            # make sure action_output is last in outputs
            fetches.append(action_output)

            outputs = sess.run(fetches=fetches, feed_dict={sy_x: x_train_batch, 
                sy_y: y_train_batch, 
                sy_l: location, 
                sy_h: state})

            correct_prediction = np.mean(np.equal(y_train_batch, outputs[-1]))
            acs.append(correct_prediction)
            ce_losses.append(outputs[9])
            pg_losses.append(outputs[10])
            path_rewards.append(outputs[8])
            if nn_baseline:
                baseline_losses.append(outputs[11])

            ######################### Print out epoch stats ########################

            print("*" * 100)
            print("Epoch: {}".format(epoch))
            print("Accuracy: {}".format(np.mean(np.array(acs))))
            print("Cross Entropy Loss: {}".format(np.mean(np.array(ce_losses))))
            print("Policy Gradient Loss: {}".format(np.mean(np.array(pg_losses))))
            if nn_baseline:
                print("Baseline Loss: {}".format(np.mean(np.array(baseline_losses))))
            print("Rewards: {}".format(np.mean(np.array(path_rewards))))


def main():
    parser = argparse.ArgumentParser()
    
    ########################## Model architecture args ##########################

    # height, width to which glimpses get resized
    parser.add_argument("--glimpse_size", type=int, default=8)
    # number of glimpses per image
    parser.add_argument("--num_glimpses", type=int, default=7)
    # number of resolutions per glimpse
    parser.add_argument("--num_resolutions", type=int, default=4) 
    # dimensionality of glimpse network output
    # TODO better names for size of glimpse image/glimpse vector
    parser.add_argument("--glimpse_vector_size", type=int, default=256)
    # dimensionality of hidden state vector
    parser.add_argument("--state_size", type=int, default=256)
    # standard deviation for Gaussian distribution over locations
    parser.add_argument("--std_dev", "-std", type=int, default=1e-3)
    # use neural network baseline
    parser.add_argument("--nn_baseline", "-bl", action="store_true")

    ############################## Training args ################################

    # number of full passes through the data
    # total training iterations = num_epochs * number of images / batch_size
    parser.add_argument("--num_epochs", type=int, default=64)
    parser.add_argument("--learning_rate", "-lr", type=int, default=1e-3)
    # batch size for each training iterations
    parser.add_argument("--batch_size", "-b", type=int, default=999)
    # random seed for deterministic training
    parser.add_argument("--random_seed", "-rs", type=int, default=42)

    ############################## Input data args ##############################

    # Defaults for these arguments are set for MNIST
    # Will need to update for DNA sequence inputs

    # dimensionality of location vector
    parser.add_argument("--loc_size", type=int, default=1)
    # original size of images
    parser.add_argument("--img_size", type=int, default=28)
    # number of classes for classification
    parser.add_argument("--num_classes", type=int, default=2)
    # number of channels in the input data
    parser.add_argument("--num_channels", type=int, default=1)
    # length of the sequences we are looking at
    parser.add_argument("--length", type=int, default=1000)
    # threshold above which examples classified as positive
    parser.add_argument("--threshold", type=float, default=0.3)
    # use this flag when running with deepsea data
    parser.add_argument("--deepsea", action="store_true")

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
        num_channels=args.num_channels,
        length=args.length,
        threshold=args.threshold,
        deepsea=args.deepsea)

if __name__ == "__main__":
    main()
