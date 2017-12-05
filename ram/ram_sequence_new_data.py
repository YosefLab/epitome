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
from time import gmtime, strftime
import glob as glob


################################# Data Processing ################################

def dna_encoder(seq, bases='ACTG'):
    # one-hot-encoding for sequence data
    # enumerates base in a sequence
    indices = [
        bases.index(x) if x in bases else -1
        for x in seq
    ]
    # one extra index for unknown
    eye = np.eye(len(bases) + 1)
    return eye[indices].astype(np.float32)


def tf_dna_encoder(seq, bases='ACTG'):
    # wraps `dna_encoder` with a `py_func`
    return tf.py_func(dna_encoder, [seq, bases], [tf.float32])[0]


def dataset_input_fn(filenames,
                     buffer_size=10000,
                     batch_size=32,
                     num_epochs=20,
                     ):
    dataset = tf.contrib.data.TFRecordDataset(filenames)

    # Use `tf.parse_single_example()` to extract data from a `tf.Example`
    # protocol buffer, and perform any additional per-record preprocessing.
    def parser(record):
        keys_to_features = {
            "sequence": tf.FixedLenFeature((), tf.string),
            "atacCounts": tf.FixedLenFeature((1000,), tf.int64),
            "Labels": tf.FixedLenFeature((1,), tf.int64),
        }
        parsed = tf.parse_single_example(record, keys_to_features)

        # Perform additional preprocessing on the parsed data.
        seq = tf_dna_encoder(parsed["sequence"])
        seq = tf.reshape(seq, [1000, 5])
        atac = parsed["atacCounts"]
        label = parsed["Labels"]

        # add more here if needed
        return {'seq': seq, 'atac': atac}, label

    # Use `Dataset.map()` to build a pair of a feature dictionary and a label
    # tensor for each example.
    dataset = dataset.map(parser)
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    iterator = dataset.make_one_shot_iterator()

    # `features` is a dictionary in which each value is a batch of values for
    # that feature; `labels` is a batch of labels.
    features, labels = iterator.get_next()
    return features, labels


##################################### CNN Code ###################################

def lrelu(x, alpha=0.2):
    # leaky-relu activation function
    return tf.maximum(alpha * x, x)


def mlp(x, n_classes):
    # specify hard-coded parameters here
    print('using mlp')

    # number of hidden units per fully-connected layer
    units = [256, 128, n_classes]
    # activation function at each layer
    activations = [lrelu, lrelu, None]

    # check that the hard-coded parameters are valid
    assert len(units) == len(activations)

    # flatten the inputs
    x = tf.contrib.layers.flatten(x)
    for u, a in zip(units, activations):
        x = tf.layers.dense(x, units=u, activation=a)

    # return logits (no activation)
    return x


def cnn(x, n_classes):
    # specify hard-coded parameters here
    print('using cnn')

    # number of filters per conv1d layer
    units = [64] * 4
    # width of the 1d kernel
    kernels = [8] * 4
    # stride at each layer
    strides = [1] * 4
    # pooling window at each layer
    pools = [2, 2, 2, 4]
    # dropout keep prob at each layer
    dropouts = [0.9] * 4
    # activation function at each layer
    activations = [lrelu] * 4

    # conv 1-4
    for u, k, s, p, d, a in zip(units,
                                kernels,
                                strides,
                                pools,
                                dropouts,
                                activations):
        x = tf.layers.conv1d(x, u, k, s,
                             padding='same',
                             activation=a)
        x = tf.layers.max_pooling1d(x, p, p,
                                    padding="same")
        x = tf.layers.dropout(x, d)

    x = tf.contrib.layers.flatten(x)

    # fc 1
    x = tf.layers.dense(x, 925)
    x = tf.contrib.layers.layer_norm(x)
    x = lrelu(x)
    x = tf.layers.dropout(x, 0.9)

    # fc 2
    x = tf.layers.dense(x, n_classes)
    x = tf.contrib.layers.layer_norm(x)
    x = tf.layers.dropout(x, 0.9)

    # return logits (no activation)
    return x


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

    location = (location/(length / 2.0)) - 1

    # assert(glimpses.shape[1] == (glimpse_size * 2) * (dna_dim + num_resolutions))
    with tf.variable_scope(scope):

        if len(glimpses.shape) == 2:
            glimpses = tf.expand_dims(glimpses, -1)
        glimpses = mlp(glimpses, n_classes=1)
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
        assert(g_t.shape[1] == output_size)
        return g_t, glimpses


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
        # assert(h_t.shape[1] == output_size)
        return out_1 + out_2


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
        assert(mean.shape[1] == 1)
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
        out = tf.layers.dense(
            out,
            units=output_size,
            activation=output_activation)
        # baseline shape should be batch_size x 1
        assert(out.shape[1] == 1)
        return out


def build_action_network(
        state,
        scope,
        output_size=10,
        n_layers=3,
        size=64,
        activation=tf.tanh,
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
    # we should never enter this loop when using deepsea_data
    glimpses = []
    for i in range(num_resolutions):
        resolution = 2**i
        glimpse = tf.nn.pool(
            input=data,
            window_shape=[resolution],
            strides=[resolution],
            pooling_type="MAX",
            padding="SAME")
        assert(glimpse.shape[1] == length / 2.0**i)
        assert(glimpse.shape[2] == 1)
        glimpses.append(glimpse)

    assert(dna.shape[1] == length)
    assert(dna.shape[2] == dna_dim)
    assert(data.shape[1] == length)
    atac_channel = 1 if num_resolutions > 0 else 0
    assert(data.shape[2] == atac_channel)
    assert(len(glimpses) == num_resolutions)

    # combine DNA and ATAC data, slice to right size, return glimpses
    return index_glimpses(dna, location, num_resolutions, glimpses, glimpse_size,
        length, batch_size, dna_dim, deepsea)


def concatenate_dna(boolean_mask, dna, glimpse_size, batch_size, dna_dim):
    padded_dna = get_padded_dna(dna, glimpse_size)

    padded_length = dna.shape[1] + (glimpse_size * 2)
    assert(padded_dna.shape[1] == padded_length)

    # get mask into correct shape, tf.stack does weird things
    dna_boolean_mask = tf.squeeze(tf.stack([boolean_mask] * dna_dim, axis=-1), axis=2)
    assert(dna_boolean_mask.shape[1] == padded_dna.shape[1])
    assert(dna_boolean_mask.shape[2] == padded_dna.shape[2])

    sliced_dna = tf.boolean_mask(tensor=padded_dna, mask=dna_boolean_mask)
    sliced_dna = tf.reshape(sliced_dna, [batch_size, glimpse_size * 2, dna_dim])
    assert(sliced_dna.shape[1] == glimpse_size * 2)
    return sliced_dna


def get_batch_glimpses(padded_glimpse, start_index, glimpse_size, batch_size):
    adder = np.arange(2*glimpse_size)
    indices = tf.map_fn(lambda index: index + adder, start_index)
    indices = tf.expand_dims(indices, -1)
    indices_index = tf.constant(np.repeat(np.arange(batch_size), glimpse_size*2).reshape([batch_size, glimpse_size*2]))
    indices_index = tf.cast(indices_index, tf.int32)
    indices_index = tf.expand_dims(indices_index, -1)
    indices = tf.concat([indices_index, indices], axis=-1)
    batched_glimpses = tf.gather_nd(tf.squeeze(padded_glimpse), indices)
    batched_glimpses = tf.expand_dims(batched_glimpses, axis=-1)
    return batched_glimpses


def index_glimpses(dna, location, num_resolutions, glimpses, glimpse_size, length, batch_size, dna_dim, deepsea):
    to_concatenate = []

    # we never enter loop with deepsea data because num_resolutions is zero
    for i in range(num_resolutions):
        glimpse = glimpses[i]
        # glimpse centered at start_index
        start_index = tf.to_int32(location / 2.0**i)
        # pad ATAC-seq and DNA with -1 values on each side
        # new length of padded data is (2 * glimpse) + length
        padded_glimpse = get_padded_glimspe(glimpse, glimpse_size)
        batched_glimpses = get_batch_glimpses(padded_glimpse, start_index, glimpse_size,
            batch_size)

        assert(glimpse.shape[1] == length / 2**i)
        assert(start_index.shape[1] == 1)
        assert(padded_glimpse.shape[1] == (length / 2**i) + (2 * glimpse_size))
        assert(padded_glimpse.shape[2] == 1)

        # concatenate DNA
        if i == 0:
            padded_dna = get_padded_dna(dna, glimpse_size)
            batched_dna_glimpses = get_batch_glimpses(tf.squeeze(padded_dna), start_index, glimpse_size, batch_size)
            # need to split the data into channels, so that the shape match ATAC
            channels = tf.split(batched_dna_glimpses, num_or_size_splits=dna_dim, axis=2)
            for channel in channels:
                to_concatenate.append(tf.squeeze(channel, axis=2))

        # TODO look into why this line caused a bug
        to_concatenate.append(batched_glimpses)

    if deepsea:
        start_index = tf.to_int32(location)
        padded_dna = get_padded_dna(dna, glimpse_size)
        batched_dna_glimpses = get_batch_glimpses(padded_dna, start_index, glimpse_size, batch_size)
        to_concatenate.append(batched_dna_glimpses)
        assert(len(to_concatenate) == 1)

    # flatten all channels
    # flat_shape = [batch_size, (num_resolutions + dna_dim) * (glimpse_size * 2)]
    # return tf.reshape(tf.concat(to_concatenate, axis=-1), flat_shape)
    not_flat_shape = [batch_size, glimpse_size * 2, num_resolutions + dna_dim]
    return tf.reshape(tf.concat(to_concatenate, axis=-1), not_flat_shape)


def get_boolean_mask(glimpse_size, start_index, length, batch_size):
    curr_index = start_index
    padded_size = length + (2 * glimpse_size)
    index_mask = tf.one_hot(indices=curr_index, depth=padded_size, axis=1)
    for i in range((glimpse_size * 2) - 1):
        curr_index += 1
        index_mask += tf.one_hot(indices=curr_index, depth=padded_size, axis=1)
    assert(curr_index.shape[1] == 1)
    assert(index_mask.shape[1] == length + 2 * glimpse_size)
    assert(index_mask.shape[2] == 1)
    return index_mask > 0


def get_padded_glimspe(glimpse, glimpse_size, constant_value=0):
    return tf.pad(glimpse, paddings=[[0, 0], [glimpse_size, glimpse_size], [0, 0]],
        constant_values=constant_value)


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
    samples = (samples + 1) * (length / 2)
    # TODO how can we avoid using tf.clip
    # clip again just to make sure indices are valid
    samples = tf.clip_by_value(samples, 0, length - 1)
    log_probs = tf.expand_dims(dist.log_prob(samples), -1)
    assert(samples.shape[1] == 1)
    assert(log_probs.shape[1] == 1)
    assert(len(log_probs.shape) == 2)
    return samples, log_probs


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
        deepsea,
        checkpoint_path):

    ################################# Download data #############################

    if deepsea:
        import deepsea_data
        # force num_resolutions to be zero, no ATAC data
        num_resolutions = 0
        dna_dim = 4
        num_classes = 919
        # TODO do not hardcode this file path
        train_batches = deepsea_data.train_iterator(
            source="../../deepsea_train/train.mat",
            batch_size=batch_size,
            num_epochs=1)
        loc_size = 1
    else:

        # data pipeline parameters
        buffer_size = 10000
        batch_size = batch_size
        num_epochs = num_epochs

        # reset the graph because it might be finalized
        # tf.reset_default_graph()

        TRAIN_PROPORTION = 0.9
        # sharded tfrecord filenames
        filenames = glob.glob('../../CEBPB-A549-hg38.txt/part-r-*')
        num_train_files = int(len(filenames) * TRAIN_PROPORTION)
        train_filenames = filenames[:num_train_files]
        valid_filenames = filenames[num_train_files:]

        features, labels = dataset_input_fn(filenames=train_filenames,
            batch_size=batch_size,
            buffer_size=buffer_size,
            num_epochs=num_epochs)

        dna = tf.cast(features['seq'], tf.float32)
        atac = tf.cast(features['atac'], tf.float32)
        chip = tf.cast(labels, tf.float32)
        x_train = tf.concat([dna, tf.expand_dims(atac, -1)], axis=-1)
        y_train = chip

        dna_dim = 5
        num_classes = 1

        # validation data
        features_valid, labels_valid = dataset_input_fn(filenames=valid_filenames,
            batch_size=batch_size,
            buffer_size=buffer_size,
            num_epochs=num_epochs)
        dna_valid = tf.cast(features_valid['seq'], tf.float32)
        atac_valid = tf.cast(features_valid['atac'], tf.float32)
        chip_valid = tf.cast(labels_valid, tf.float32)
        x_valid = tf.concat([dna_valid, tf.expand_dims(atac_valid, -1)], axis=-1)
        y_valid = chip_valid


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

    glimpse_output, glimpses = build_glimpse_network(
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

    assert(glimpse_output.shape[1] == glimpse_vector_size)
    assert(hidden_output.shape[1] == state_size)
    assert(baseline_output.shape[1] == 1)
    assert(raw_action_output.shape[1] == num_classes)
    assert(sigmoid_output.shape[1] == num_classes)
    assert(action_output.shape[1] == num_classes)
    assert(mean_location_output.shape[1] == 1)
    assert(location_output.shape[1] == 1)
    assert(log_probs.shape[1] == 1)

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
        assert(baseline_prediction.shape[1] == 1)
    else:
        adv_n = rewards

    policy_gradient_loss = tf.scalar_mul(-1, tf.reduce_mean(log_probs * tf.to_float(adv_n)))

    if nn_baseline:
        baseline_loss = tf.losses.mean_squared_error(rewards, baseline_prediction)
        hybrid_loss = cross_entropy_loss + policy_gradient_loss + baseline_loss
    else:
        hybrid_loss = cross_entropy_loss + policy_gradient_loss

    update_op = tf.train.AdamOptimizer(learning_rate).minimize(hybrid_loss)

    assert(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.cast(sy_y, tf.float32),
        logits=raw_action_output).shape[1] == num_classes)
    assert(rewards.shape[1] == 1)
    assert(adv_n.shape[1] == 1)

    # computing area under ROC
    sy_auc, auc_op = tf.metrics.auc(
        labels=sy_y,
        predictions=sigmoid_output)

    ############################## Tensorflow engineering #######################

    # # initialize config
    # tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
    #
    # # initialize session and variables
    # sess = tf.Session(config=tf_config)
    # sess.__enter__()  # equivalent to `with sess:`
    # tf.global_variables_initializer().run()  # pylint: disable=E1101
    # # create saver to checkpoint model
    # saver = tf.train.Saver()

    ################################### Train ###################################

    with tf.train.MonitoredSession() as sess:

        # accuracies
        acs = []
        # cross entropy and policy gradient losses
        ce_losses = []
        pg_losses = []
        baseline_losses = []
        # total rewards for num_glimpses timesteps
        path_rewards = []

        iter = 0
        while not sess.should_stop():

            x_train_batch, y_train_batch = sess.run([x_train, y_train])

            # TODO find a more elegant way to consider last bit of data
            print(iter)
            if len(x_train_batch) < batch_size:
                continue

            # hidden state initialized to zeros
            state = np.zeros(shape=[x_train_batch.shape[0], state_size])
            # location = np.random.randint(0, length, size=[x_train_batch.shape[0], loc_size])
            location = np.zeros(shape=[x_train_batch.shape[0], loc_size]) + (length/2.0)

            for j in range(num_glimpses - 1):
                fetches = [location_output, hidden_output, glimpse_output]
                outputs = sess.run(fetches=fetches, feed_dict={sy_x: x_train_batch,
                    sy_y: y_train_batch,
                    sy_l: location,
                    sy_h: state})

                location = outputs[0]
                state = outputs[1]
                # print(outputs[2])

            fetches = [update_op, glimpse_output, location_output, mean_location_output, log_probs, hidden_output,
                       raw_action_output, sigmoid_output, rewards, cross_entropy_loss, policy_gradient_loss, glimpses,
                       sy_auc, auc_op]

            if nn_baseline:
                fetches.append(baseline_loss)
            # make sure action_output is last in outputs
            fetches.append(action_output)

            outputs = sess.run(fetches=fetches, feed_dict={sy_x: x_train_batch,
                sy_y: y_train_batch,
                sy_l: location,
                sy_h: state})

            correct_prediction = np.mean(np.equal(y_train_batch, outputs[-1]))
            assert(np.equal(y_train_batch, outputs[-1]).shape[1] == num_classes)

            acs.append(correct_prediction)
            ce_losses.append(outputs[9])
            pg_losses.append(outputs[10])
            path_rewards.append(outputs[8])
            if nn_baseline:
                baseline_losses.append(outputs[12])


            ######################### Print out epoch stats ########################

            print("*" * 100)
            print("Iteration: {}".format(iter))
            print("Accuracy: {}".format(correct_prediction))
            print("Cross Entropy Loss: {}".format(outputs[9]))
            print("Policy Gradient Loss: {}".format(outputs[10]))
            print("AUC: {}".format(outputs[12]))

            # compute validation data stats
            x_valid_batch, y_valid_batch = sess.run([x_valid, y_valid])
            if len(x_valid_batch) < batch_size:
                continue
            fetches_valid = [sy_auc, auc_op, action_output, cross_entropy_loss, policy_gradient_loss]
            outputs_valid = sess.run(fetches=fetches_valid,
                                     feed_dict={sy_x: x_valid_batch,
                                               sy_y: y_valid_batch,
                                               sy_l: location,
                                               sy_h: state})

            correct_prediction_valid = np.mean(np.equal(y_valid_batch, outputs_valid[2]))

            print("Validation AUC: {}".format(outputs_valid[0]))
            print("Validation accuracy: {}".format(correct_prediction_valid))

            # if nn_baseline:
            #     print("Baseline Loss: {}".format(outputs[12]))
            # print("Rewards: {}".format(np.mean(np.array(path_rewards))))
            # print(outputs[7][:10])
            # print(outputs[2][:10])
            # if iter > 10:
            #     import pdb; pdb.set_trace()


            ############################ Save the model ############################

            # save after each batch
            # saver.save(sess, checkpoint_path + strftime("%Y-%m-%d--%H:%M:%S", gmtime()))

            iter +=1

        # save the final model
        # saver.save(sess, checkpoint_path + strftime("%Y-%m-%d--%H:%M:%S--final", gmtime()))


def main():
    parser = argparse.ArgumentParser()

    ########################### Model architecture args ############################

    # height, width to which glimpses get resized
    parser.add_argument("--glimpse_size", type=int, default=50)
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
    parser.add_argument("--std_dev", "-std", type=float, default=1.0)
    # use neural network baseline
    parser.add_argument("--nn_baseline", "-bl", action="store_true")

    ############################## Training args ################################

    # number of full passes through the data
    # total training iterations = num_epochs * number of images / batch_size
    parser.add_argument("--num_epochs", type=int, default=64)
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-3)
    # batch size for each training iterations
    parser.add_argument("--batch_size", "-b", type=int, default=1000)
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
    parser.add_argument("--num_classes", type=int, default=1)
    # number of channels in the input data
    parser.add_argument("--num_channels", type=int, default=1)
    # length of the sequences we are looking at
    parser.add_argument("--length", type=int, default=1000)
    # threshold above which examples classified as positive
    parser.add_argument("--threshold", type=float, default=0.5)
    # use this flag when running with deepsea data
    parser.add_argument("--deepsea", action="store_true")
    # path to which models checkpoint gets saved
    parser.add_argument("--path", type=str, default="/tmp/")

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
        deepsea=args.deepsea,
        checkpoint_path=args.path)

if __name__ == "__main__":
    main()
