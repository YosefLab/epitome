"""
Model specifications here.
"""
import tensorflow as tf
import numpy as np


def lrelu(x, alpha=0.2):
    """Leaky ReLU activation.

    Args:
        x: A tensor.
        alpha: Float. For piecewise linear composition.
    Returns:
        A tensor with the Leaky ReLU activation applied elementwise.
    """
    return tf.maximum(alpha*x, x)


def fc(x, n_units, dropout, activation=None):
    """Fully connected layer with dropout.

    Args:
        x: Tensor with shape [batch, dimensions].
        n_units: Int. Number of output units.
        dropout: Float. Probability to keep an activation.
        activation: An activation function. If None, linear activation.

    Returns:
        The output activation of the layer.
    """
    net = tf.layers.dense(x, n_units)
    net = tf.contrib.layers.layer_norm(net)
    if activation:
        net = activation(net)
    return tf.layers.dropout(net, dropout)


def conv1d(x, hidden_size, kernel_size, stride=1, dilation=1,
           pooling_size=0, dropout=0.0, activation=None):
    """A convolutional layer.

    Args:
        x: Tensor with shape [batch, length, dim]
        hidden_size: Int. The number of filters in this layer.
        kernel_size: Int. The width of the kernels in this layer.
        stride: Int. How much to stride the filters.
        dilation: Int. How much to dilate the filters.
        pooling_size: Int. How much to pool the outputs of the conv layer.
        dropout: Float. Probability to keep an activation.
        activation: An activation function. If None, linear activation.

    Returns:
        The output activation of the layer.
    """
    net = tf.layers.conv1d(x, hidden_size, kernel_size, stride, padding='same',
                           dilation_rate=dilation, activation=activation)
    if pooling_size:
        net = tf.layers.max_pooling1d(net, pooling_size, pooling_size,
                                      padding="same")
    return tf.layers.dropout(net, dropout)


def cnn(input_, n_classes, hp):
    """Contructs a 1D convolutional neural network from a set of hparams.

    Arguments:
        input_: Input to the model, should be shape [batch, length, dim].
        h_classes: Int. Number of dimensions in the logits.
        hp: A hyperparameter set to be used to wire the network.
            There should be a set of defaults one can build from below.

    Returns:
        The predictions of the model.
    """
    net = input_
    for i in range(hp.n_conv_layers):
        net = conv1d(net, hp.hidden_sizes[i], hp.kernel_size, hp.stride,
                     dilation=1, pooling_size=hp.pooling_sizes[i],
                     dropout=hp.dropout_keep_probs[i], activation=hp.activation)
    for i in range(hp.n_dconv_layers):
        dilation= 2**(i + 1)
        tmp = conv1d(net, hp.dconv_h_size, hp.kernel_size, hp.stride,
                     dilation=1, pooling_size=0, dropout=hp.dropout,
                     activation=hp.activation)
        net = tf.concat([net, tmp], axis=2)
    net = tf.contrib.layers.flatten(net)
    net = fc(net, hp.fc_h_size, hp.dropout, activation=hp.activation)
    return fc(net, n_classes, hp.dropout, activation=hp.output_activation)


def cheating_cnn(input_, dnase, n_classes, hp):
    """Contructs a 1D convolutional neural network from a set of hparams.

    This network cheats! It differs from cnn(...) by TODO(weston).

    Arguments:
        input_: Input to the model, should be shape [batch, length, dim].
        h_classes: Int. Number of dimensions in the logits.
        hp: A hyperparameter set to be used to wire the network.
            There should be a set of defaults one can build from below.

    Returns:
        The predictions of the model.
    """
    net = input_
    for i in range(hp.n_conv_layers):
        net = conv1d(net, hp.hidden_sizes[i], hp.kernel_size, hp.stride,
                     dilation=1, pooling_size=hp.pooling_sizes[i],
                     dropout=hp.dropout_keep_probs[i], activation=hp.activation)
    for i in range(hp.n_dconv_layers):
        dilation= 2**(i + 1)
        tmp = conv1d(net, hp.dconv_h_size, hp.kernel_size, hp.stride,
                     dilation=1, pooling_size=0, dropout=hp.dropout,
                     activation=hp.activation)
        net = tf.concat([net, tmp], axis=2)
    net = tf.contrib.layers.flatten(net)
    net = tf.concat([net, dnase], axis = 1)
    net = fc(net, hp.fc_h_size, hp.dropout, activation=hp.activation)
    return fc(net, n_classes, hp.dropout, activation=hp.output_activation)


def cheating_cnn2(input_, dnase, n_classes, hp):
    """Contructs a 1D convolutional neural network from a set of hparams.

    This network cheats! It differs from chearing_cnn(...) by TODO(weston).

    Arguments:
        input_: Input to the model, should be shape [batch, length, dim].
        h_classes: Int. Number of dimensions in the logits.
        hp: A hyperparameter set to be used to wire the network.
            There should be a set of defaults one can build from below.

    Returns:
        The predictions of the model.
    """
    net = input_
    for i in range(hp.n_conv_layers):
        net = conv1d(net, hp.hidden_sizes[i], hp.kernel_size, hp.stride,
                     dilation=1, pooling_size=hp.pooling_sizes[i],
                     dropout=hp.dropout_keep_probs[i], activation=hp.activation)
    for i in range(hp.n_dconv_layers):
        dilation= 2**(i + 1)
        tmp = conv1d(net, hp.dconv_h_size, hp.kernel_size, hp.stride,
                     dilation=1, pooling_size=0, dropout=hp.dropout,
                     activation=hp.activation)
        net = tf.concat([net, tmp], axis=2)
    net = tf.contrib.layers.flatten(net)
    net = tf.concat([net, dnase], axis = 1)
    net = fc(net, hp.fc_h_size, hp.dropout, activation=hp.activation)
    net = fc(net, hp.fc_h_size, hp.dropout, activation=hp.activation)
    return fc(net, n_classes, hp.dropout, activation=hp.output_activation)


def rnn(input_, n_classes, hp): # RNN
    """Contructs a recurrent neural network (RNN) from a set of hparams.

    Arguments:
        input_: Input to the model, should be shape [batch, length, dim].
        h_classes: Int. Number of dimensions in the logits.
        hp: A hyperparameter set to be used to wire the network.
            There should be a set of defaults one can build from below.

    Returns:
        The predictions of the model.
    """
    cell = tf.contrib.rnn.BasicLSTMCell(hp.hidden_size, forget_bias=1.0)
    
    values = tf.unstack(input_, axis=1)
    outputs, states = tf.contrib.rnn.static_rnn(cell, values, dtype=tf.float32)
    return fc(outputs[-1], n_classes, hp.dropout,
              activation=hp.output_activation)


def srnn(input_, n_classes, hp): # Stacked RNN
    """Contructs a stacked RNN from a set of hparams.

    Arguments:
        input_: Input to the model, should be shape [batch, length, dim].
        h_classes: Int. Number of dimensions in the logits.
        hp: A hyperparameter set to be used to wire the network.
            There should be a set of defaults one can build from below.

    Returns:
        The predictions of the model.
    """
    cells = [tf.contrib.rnn.BasicLSTMCell(hp.hidden_size, forget_bias=1.0)
             for _ in range(hp.n_layers)]
    cell = tf.contrib.rnn.MultiRNNCell(cells)
    
    values = tf.unstack(input_, axis=1)
    outputs, states = tf.contrib.rnn.static_rnn(cell, values, dtype=tf.float32)
    return fc(outputs[-1], n_classes, hp.dropout,
              activation=hp.output_activation)


def birnn(input_, n_classes, hp):
    """Contructs a bidirectional RNN from a set of hparams.

    Arguments:
        input_: Input to the model, should be shape [batch, length, dim].
        h_classes: Int. Number of dimensions in the logits.
        hp: A hyperparameter set to be used to wire the network.
            There should be a set of defaults one can build from below.

    Returns:
        The predictions of the model.
    """
    fw = tf.contrib.rnn.BasicLSTMCell(hp.hidden_size, forget_bias=1.0)
    bw = tf.contrib.rnn.BasicLSTMCell(hp.hidden_size, forget_bias=1.0)
    
    values = tf.unstack(input_, axis=1)
    outputs, fw_states, bw_states = tf.contrib.rnn.static_bidirectional_rnn(
        fw, bw, values, dtype=tf.float32)
    return fc(outputs[-1], n_classes, hp.dropout, activation=hp.output_activation)


def cnn_hp(**kwargs):
    """Constructs a default set of hyperparameters for a CNN.

    Args:
        kwargs: keyword arguments to override defaults.

    Returns:
        An HParam object to construct a CNN.
    """
    hp = tf.contrib.training.HParams()
    hp.n_conv_layers = 4
    hp.n_dconv_layers = 0
    hp.hidden_sizes = [128,128,128,64]
    hp.dconv_h_size = 64
    hp.fc_h_size = 925
    hp.kernel_size = 8
    hp.pooling_sizes = [2, 2, 2, 4]
    hp.stride = 1
    hp.dropout_keep_probs = [0.9, 0.9, 0.9, 0.9]
    hp.dropout = 1
    hp.activation = lrelu
    hp.output_activation = tf.sigmoid
    hp.__dict__.update(kwargs)
    return hp

def rnn_hp(**kwargs):
    """Constructs a default set of hyperparameters for a RNN.

    Args:
        kwargs: keyword arguments to override defaults.

    Returns:
        An HParam object to construct a RNN.
    """
    hp = tf.contrib.training.HParams()
    hp.hidden_size = 64
    hp.n_layers = 2 # Stacked RNN
    hp.dropout = 0.9
    hp.output_activation = tf.sigmoid
    hp.__dict__.update(kwargs)
    return hp

def build_CNN_graph(DNAse = False, pos_weight = 50, rate = 1e-3, hp = cnn_hp()):
    """Builds a CNN graph.

    TODO(weston): stop mixing capitals and underscores and other gross stuff.
        Also no spaces between equal signs when specifying argument defaults.

    TODO(alex): stop introducing bugs into the code.

    Args:
        DNAse: Boolean. Whether or not to use DNAse as input to the network.
        pos_weight: Float. How much to weight positive examples.
        rate: Float. Initial learning rate to the model.
        hp: HParam set to configure the CNN.

    Returns:
        A dictionary of Tensors to be fed into the main training loop.
        See `main(...)` in `main.py` for usage.
    """
    num_logits = 816 - 126

    input_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, 1000, 4])
    dnase_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, 126])
    target_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, num_logits])
    if DNAse:
        logits = cheating_cnn(input_placeholder, dnase_placeholder, num_logits, hp)
    else:
        logits = cnn(input_placeholder, num_logits, hp)
    loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
        logits=logits,targets=target_placeholder, pos_weight=pos_weight))
    optimizer = tf.train.AdamOptimizer(rate).minimize(loss)
        
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()

    return {"input_placeholder": input_placeholder,
     		"dnase_placeholder": dnase_placeholder,
     		"target_placeholder": target_placeholder,
     		"optimizer": optimizer,
    		"logits": logits,
    		"loss": loss,
    		"init_op": init_op,
     		"saver": saver}