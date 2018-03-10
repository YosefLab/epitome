"""
Model specifications here.
"""
import tensorflow as tf
import numpy as np
import codecs
import json


################################# LAYERS #######################################


def lrelu(x, alpha=0.2):
    """Leaky ReLU activation.

    Args:
        x: A tensor.
        alpha: Float. For piecewise linear composition.
    Returns:
        A tensor with the Leaky ReLU activation applied elementwise.
    """
    return tf.maximum(alpha*x, x)


def fc(x, n_units, dropout, activation=None, training=False, l1=0.0):
    """Fully connected layer with dropout.

    Args:
        x: Tensor with shape [batch, dimensions].
        n_units: Int. Number of output units.
        dropout: Float. Probability to keep an activation.
        activation: An activation function. If None, linear activation.

    Returns:
        The output activation of the layer.
    """
    print(l1)
    net = tf.layers.dense(x, n_units, 
        kernel_regularizer=tf.contrib.layers.l1_regularizer(l1))
    net = tf.contrib.layers.layer_norm(net)
    if activation:
        net = activation(net)
    return tf.layers.dropout(
        net, dropout, training=training)


def conv1d(x, hidden_size, kernel_size, stride=1, dilation=1,
           pooling_size=0, dropout=0.0, activation=None, gated=False,
           training=False, l2=0.0):
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
                           dilation_rate=dilation, activation=activation,
                           kernel_regularizer=
                           tf.contrib.layers.l2_regularizer(l2))
    if gated:
        # Gated linear wrapper around conv1d.
        # https://arxiv.org/pdf/1612.08083.pdf
        gate = tf.layers.conv1d(x, hidden_size, kernel_size, stride,
                                padding='same', dilation_rate=dilation,
                                activation=tf.nn.sigmoid)
        net = tf.multiply(net, gate)
    if pooling_size:
        net = tf.layers.max_pooling1d(net, pooling_size, pooling_size,
                                      padding="same")
    return tf.layers.dropout(net, dropout, training=training)


################################# MODELS #######################################


def cnn(input_, n_classes, hp, training=False):
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
                     dropout=hp.drop_probs[i], activation=hp.activation,
                     training=training, gated=hp.gated, l2=hp.l2)
    for i in range(hp.n_dconv_layers):
        dilation= 2**(i + 1)
        tmp = conv1d(net, hp.dconv_h_size, hp.kernel_size, hp.stride,
                     dilation=1, pooling_size=0, dropout=hp.dropout,
                     activation=hp.activation, training=training,
                     gated=hp.gated, l2=hp.l2)
        net = tf.concat([net, tmp], axis=2)
    net = tf.contrib.layers.flatten(net)
    net = fc(net, hp.fc_h_size, hp.dropout, activation=hp.activation, 
        training=training, l1=hp.l1)
    return fc(net, n_classes, hp.dropout, activation=hp.output_activation,
        training=training, l1=hp.l1)


def cheating_cnn(input_, dnase, n_classes, hp, training=False):
    """ Depricated: When using DNase, use tfrecords and a regular CNN.

    Contructs a 1D convolutional neural network from a set of hparams.

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
                     dropout=hp.drop_probs[i], activation=hp.activation,
                     training=training)
    for i in range(hp.n_dconv_layers):
        dilation= 2**(i + 1)
        tmp = conv1d(net, hp.dconv_h_size, hp.kernel_size, hp.stride,
                     dilation=1, pooling_size=0, dropout=hp.dropout,
                     activation=hp.activation, training=training)
        net = tf.concat([net, tmp], axis=2)
    net = tf.contrib.layers.flatten(net)
    net = tf.concat([net, dnase], axis = 1)
    net = fc(net, hp.fc_h_size, hp.dropout, activation=hp.activation,
        training=training)
    return fc(net, n_classes, hp.dropout, activation=hp.output_activation,
        training=training)


def cheating_cnn2(input_, dnase, n_classes, hp, training=False):
    """ Depricated: When using DNase, use tfrecords and a regular CNN.

    Contructs a 1D convolutional neural network from a set of hparams.

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
                     dropout=hp.drop_probs[i], activation=hp.activation,
                     training=training)
    for i in range(hp.n_dconv_layers):
        dilation= 2**(i + 1)
        tmp = conv1d(net, hp.dconv_h_size, hp.kernel_size, hp.stride,
                     dilation=1, pooling_size=0, dropout=hp.dropout,
                     activation=hp.activation, training=training)
        net = tf.concat([net, tmp], axis=2)
    net = tf.contrib.layers.flatten(net)
    net = tf.concat([net, dnase], axis = 1)
    net = fc(net, hp.fc_h_size, hp.dropout, activation=hp.activation,
        training=training)
    net = fc(net, hp.fc_h_size, hp.dropout, activation=hp.activation,
        training=training)
    return fc(net, n_classes, hp.dropout, activation=hp.output_activation,
        training=training)


def github_pwms_cnn(input_, pwms, n_classes, hp, training=False):
    """Convolutional network using PWMs as fixed filters from the first layer.
    Same architecture as the OrbWeaver model.

    Arguments:
        input_: Input to the model, should be shape [batch, length, dim].
        pwms: Position weight matrices for first layer filters.
        n_classes: Int. Number of dimensions in the logits.
        hp: A hyperparameter set to be used to wire the network.
            There should be a set of defaults one can build from below.

    Returns:
        The predictions of the model.
    """
    net = input_
    net = tf.nn.conv2d(
        input=net,
        filter=pwms,
        strides=[1, 1, 1, 1],
        padding="VALID",
        # dilations=[1, 1, 1, 1],
        name=None)
    
    net = lrelu(net)
    
    net = tf.layers.max_pooling2d(
        inputs=net, 
        pool_size=(4,1), 
        strides=(4,1), 
        padding="valid")

    net = tf.layers.conv2d(
        inputs=net,
        filters=200,
        kernel_size=(6, 1),
        strides=(1, 1),
        padding="valid",
        dilation_rate=(1, 1),
        activation=tf.nn.relu,
        kernel_initializer=tf.glorot_uniform_initializer())
    
    net = tf.layers.max_pooling2d(
        inputs=net, 
        pool_size=(3, 1), 
        strides=(3, 1), 
        padding="valid")
    
    net = tf.contrib.layers.flatten(net)

    net = tf.layers.dense(
        net,
        units=500,
        activation=tf.nn.relu,
        kernel_initializer=tf.glorot_uniform_initializer())
    
    # TODO check if this should be gaussian noise
    net = tf.layers.dropout(net, rate=0.8, training=training)

    net = tf.layers.dense(
        net,
        units=n_classes,
        activation=hp.output_activation,
        kernel_initializer=tf.glorot_uniform_initializer())
    return net


def our_pwms_cnn(input_, pwms, n_classes, hp, training=False):
    """Convolutional network using PWMs as fixed filters from the first layer.
    Only the first layer of the network is modeled after OrbWeaver. 
    Rest of architecture is modeled after cnn().

    Arguments:
        input_: Input to the model, should be shape [batch, length, dim].
        pwms: Position weight matrices for first layer filters.
        h_classes: Int. Number of dimensions in the logits.
        hp: A hyperparameter set to be used to wire the network.
            There should be a set of defaults one can build from below.

    Returns:
        The predictions of the model.
    """
    net = input_
    net = tf.nn.conv2d(
        input=net,
        filter=pwms,
        strides=[1, 1, 1, 1],
        padding="VALID",
        # dilations=[1, 1, 1, 1],
        name=None)

    net = lrelu(net)
    
    net = tf.layers.max_pooling2d(
        inputs=net, 
        pool_size=(4,1), 
        strides=(4,1), 
        padding="valid")
    net = tf.squeeze(net, 2)

    # skip first conv layer since we used PWMs
    for i in range(1, hp.n_conv_layers):
        net = conv1d(net, hp.hidden_sizes[i], hp.kernel_size, hp.stride,
                     dilation=1, pooling_size=hp.pooling_sizes[i],
                     dropout=hp.drop_probs[i], activation=hp.activation,
                     training=training)
    for i in range(hp.n_dconv_layers):
        dilation= 2**(i + 1)
        tmp = conv1d(net, hp.dconv_h_size, hp.kernel_size, hp.stride,
                     dilation=1, pooling_size=0, dropout=hp.dropout,
                     activation=hp.activation, training=training)
        net = tf.concat([net, tmp], axis=2)
    net = tf.contrib.layers.flatten(net)
    net = fc(net, hp.fc_h_size, hp.dropout, activation=hp.activation, 
        training=training)
    return fc(net, n_classes, hp.dropout, activation=hp.output_activation,
        training=training)


def rnn(input_, n_classes, hp, training=False): # RNN
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
              activation=hp.output_activation, training=training)


def srnn(input_, n_classes, hp, training=False): # Stacked RNN
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
              activation=hp.output_activation, training=training)


def birnn(input_, n_classes, hp, training=False):
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
    return fc(outputs[-1], n_classes, hp.dropout,
     activation=hp.output_activation, training=training)


############################## DEFAULT HPARAMS #################################


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
    hp.drop_probs = [0., 0., 0., 0.]
    hp.dropout = 0.
    hp.activation = lrelu
    hp.output_activation = tf.sigmoid
    hp.gated = False
    hp.l1 = 0.0
    hp.l2 = 0.0
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


############################## HPARAMS UTILS ###################################


def save_hparams(hparams_file, hparams):
  """Save hparams."""
  with codecs.getwriter("utf-8")(tf.gfile.GFile(hparams_file, "wb")) as f:
    f.write(hparams.to_json())


def load_hparams(hparams_file):
  """Load hparams from an existing model directory."""
  if tf.gfile.Exists(hparams_file):
    with codecs.getreader("utf-8")(tf.gfile.GFile(hparams_file, "rb")) as f:
      try:
        hparams_values = json.load(f)
        print(hparams_values)
        hparams = tf.contrib.training.HParams(**hparams_values)
      except ValueError:
        print_out("  can't load hparams file")
        return None
    return hparams
  else:
    return None


def parse_hparams_string(string):
    """Parses a string specifying hparams.
    Examples:
        `n_conv_layers=10,gated=True`
        `dropout=10,fc_h_size=1024`
    """
    if string is None:
        return {}
    tuples = [s.split('=') for s in string.split('/')]
    return {k: eval(v) for k, v in tuples}


############################## GRAPH UTILS #####################################


def build_cnn_graph(DNAse=False, pos_weight=50, hp=cnn_hp(), 
    tfrecords=False, num_logits=815-125):
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
    if tfrecords:
        in_height = 5
    else:
        in_height = 4

    input_placeholder = tf.placeholder(dtype=tf.float32,
     shape=[None, 1000, in_height])
    dnase_placeholder = tf.placeholder(dtype=tf.float32,
     shape=[None, 125])
    target_placeholder = tf.placeholder(dtype=tf.float32, 
     shape=[None, num_logits])
    mask_default = tf.constant(1., shape=[1, num_logits])
    mask_placeholder = tf.placeholder_with_default(mask_default, 
     shape=[None, num_logits])
    training = tf.placeholder(dtype = tf.bool)
    rate = tf.placeholder(dtype=tf.float32)

    if DNAse and not tfrecords:
        logits = cheating_cnn(input_placeholder, dnase_placeholder, num_logits,
         hp, training)
    else:
        logits = cnn(input_placeholder, num_logits, hp, training)
        logits = tf.multiply(logits, mask_placeholder)

    loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
            logits=logits,targets=target_placeholder, pos_weight=pos_weight))
    if hp.l1 or hp.l2:
        loss += tf.losses.get_regularization_losses()

    optimizer = tf.train.AdamOptimizer(learning_rate=rate).minimize(loss)
        
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()

    return {"input_placeholder": input_placeholder,
     		"dnase_placeholder": dnase_placeholder,
     		"target_placeholder": target_placeholder,
     		"optimizer": optimizer,
    		"logits": logits,
    		"loss": loss,
    		"init_op": init_op,
            "training": training,
     		"saver": saver,
            "rate": rate}
