import tensorflow as tf
import numpy as np


def lrelu(x, alpha=0.2):
    return tf.maximum(alpha*x, x)


def fc(x, n_units, dropout, activation=None):
    net = tf.layers.dense(x, n_units)
    net = tf.contrib.layers.layer_norm(net)
    if activation:
        net = activation(net)
    return tf.layers.dropout(net, dropout)


def conv1d(x, hidden_size, kernel_size, stride=1, dilation=1,
           pooling_size=0, dropout=0.0, activation=None):
    net = tf.layers.conv1d(x, hidden_size, kernel_size, stride, padding='same',
                           dilation_rate=dilation, activation=activation)
    if pooling_size:
        net = tf.layers.max_pooling1d(net, pooling_size, pooling_size, padding="same")
    return tf.layers.dropout(net, dropout)


def cnn(input_, n_classes, hp):
    net = input_
    for i in range(hp.n_conv_layers):
        net = conv1d(net, hp.hidden_sizes[i], hp.kernel_size, hp.stride, dilation=1,
                     pooling_size=hp.pooling_sizes[i], dropout=hp.dropout_keep_probs[i],
                     activation=hp.activation)
    for i in range(hp.n_dconv_layers):
        dilation= 2**(i + 1)
        tmp = conv1d(net, hp.dconv_h_size, hp.kernel_size, hp.stride, dilation=1,
                     pooling_size=0, dropout=hp.dropout, activation=hp.activation)
        net = tf.concat([net, tmp], axis=2)
    net = tf.contrib.layers.flatten(net)
    net = fc(net, hp.fc_h_size, hp.dropout, activation=hp.activation)
    return fc(net, n_classes, hp.dropout, activation=hp.output_activation)


def cheating_cnn(input_, dnase, n_classes, hp):
    net = input_
    for i in range(hp.n_conv_layers):
        net = conv1d(net, hp.hidden_sizes[i], hp.kernel_size, hp.stride, dilation=1,
                     pooling_size=hp.pooling_sizes[i], dropout=hp.dropout_keep_probs[i],
                     activation=hp.activation)
    for i in range(hp.n_dconv_layers):
        dilation= 2**(i + 1)
        tmp = conv1d(net, hp.dconv_h_size, hp.kernel_size, hp.stride, dilation=1,
                     pooling_size=0, dropout=hp.dropout, activation=hp.activation)
        net = tf.concat([net, tmp], axis=2)
    net = tf.contrib.layers.flatten(net)
    net = tf.concat([net, dnase], axis = 1)
    net = fc(net, hp.fc_h_size, hp.dropout, activation=hp.activation)
    return fc(net, n_classes, hp.dropout, activation=hp.output_activation)


def cheating_cnn2(input_, dnase, n_classes, hp):
    net = input_
    for i in range(hp.n_conv_layers):
        net = conv1d(net, hp.hidden_sizes[i], hp.kernel_size, hp.stride, dilation=1,
                     pooling_size=hp.pooling_sizes[i], dropout=hp.dropout_keep_probs[i],
                     activation=hp.activation)
    for i in range(hp.n_dconv_layers):
        dilation= 2**(i + 1)
        tmp = conv1d(net, hp.dconv_h_size, hp.kernel_size, hp.stride, dilation=1,
                     pooling_size=0, dropout=hp.dropout, activation=hp.activation)
        net = tf.concat([net, tmp], axis=2)
    net = tf.contrib.layers.flatten(net)
    net = tf.concat([net, dnase], axis = 1)
    net = fc(net, hp.fc_h_size, hp.dropout, activation=hp.activation)
    net = fc(net, hp.fc_h_size, hp.dropout, activation=hp.activation)
    return fc(net, n_classes, hp.dropout, activation=hp.output_activation)


def rnn(input_, n_classes, hp): # RNN
    cell = tf.contrib.rnn.BasicLSTMCell(hp.hidden_size, forget_bias=1.0)
    
    values = tf.unstack(input_, axis=1)
    outputs, states = tf.contrib.rnn.static_rnn(cell, values, dtype=tf.float32)
    return fc(outputs[-1], n_classes, hp.dropout, activation=hp.output_activation)


def srnn(input_, n_classes, hp): # Stacked RNN
    cells = [tf.contrib.rnn.BasicLSTMCell(hp.hidden_size, forget_bias=1.0)
             for _ in range(hp.n_layers)]
    cell = tf.contrib.rnn.MultiRNNCell(cells)
    
    values = tf.unstack(input_, axis=1)
    outputs, states = tf.contrib.rnn.static_rnn(cell, values, dtype=tf.float32)
    return fc(outputs[-1], n_classes, hp.dropout, activation=hp.output_activation)


def birnn(input_, n_classes, hp): # Bidirectional RNN
    fw = tf.contrib.rnn.BasicLSTMCell(hp.hidden_size, forget_bias=1.0)
    bw = tf.contrib.rnn.BasicLSTMCell(hp.hidden_size, forget_bias=1.0)
    
    values = tf.unstack(input_, axis=1)
    outputs, fw_states, bw_states = tf.contrib.rnn.static_bidirectional_rnn(
        fw, bw, values, dtype=tf.float32)
    return fc(outputs[-1], n_classes, hp.dropout, activation=hp.output_activation)

def cnn_hp(**kwargs):
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
    hp = tf.contrib.training.HParams()
    hp.hidden_size = 64
    hp.n_layers = 2 # Stacked RNN
    hp.dropout = 0.9
    hp.output_activation = tf.sigmoid
    hp.__dict__.update(kwargs)
    return hp

def build_CNN_graph(DNAse = False, pos_weight = 50, rate = 1e-3, hp = cnn_hp()):
    input_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, 1000, 4])
    dnase_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, 126])
    target_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, 919 - 126])
    if DNAse:
        logits = cheating_cnn(input_placeholder, dnase_placeholder, 919 - 126, hp)
    else:
        logits = cnn(input_placeholder, 919 - 126, hp)
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


