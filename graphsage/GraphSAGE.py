# -*- coding:utf-8 -*-
# @Time : 2021/9/7 11:29 下午
# @Author : huichuan LI
# @File : GraphSAGE.py
# @Software: PyCharm

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.initializers import glorot_uniform, Zeros
from tensorflow.python.keras.layers import Input, Dense, Dropout, Layer, LSTM
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2


class MeanAggregator(Layer):
    def __init__(self, units, input_dim, neigh_max, concat=True, dropout_rate=0.0, activation=tf.nn.relu, l2_reg=0,
                 use_bias=False, gcn=True,
                 seed=1024, **kwargs):
        super(MeanAggregator, self).__init__()
        self.units = units
        self.neigh_max = neigh_max
        self.concat = concat
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.use_bias = use_bias
        self.activation = activation
        self.seed = seed
        self.input_dim = input_dim
        self.gcn = gcn

    def build(self, input_shapes):
        if self.gcn:
            self.neigh_weights = self.add_weight(shape=(self.input_dim * 2, self.units),
                                                 initializer=glorot_uniform(
                                                     seed=self.seed),
                                                 regularizer=l2(self.l2_reg),
                                                 name="neigh_weights")

        else:
            self.neigh_weights = self.add_weight(shape=(self.input_dim, self.units),
                                                 initializer=glorot_uniform(
                                                     seed=self.seed),
                                                 regularizer=l2(self.l2_reg),
                                                 name="neigh_weights")
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units), initializer=Zeros(),
                                        name='bias_weight')
        self.dropout = Dropout(self.dropout_rate)
        self.built = True

    def call(self, inputs, training=None):
        features, node, neighbours = inputs

        node_feat = tf.nn.embedding_lookup(features, node)
        neigh_feat = tf.nn.embedding_lookup(features, neighbours)
        print(node_feat)
        print(neigh_feat)

        node_feat = self.dropout(node_feat, training=training)
        neigh_feat = self.dropout(neigh_feat, training=training)
        if self.gcn:
            concat_mean = tf.reduce_mean(neigh_feat, axis=1, keepdims=False)
            concat_feat = tf.concat([tf.squeeze(node_feat), concat_mean], axis=1)
            output = tf.matmul(concat_feat, self.neigh_weights)
        else:
            concat_mean = tf.reduce_mean(neigh_feat, axis=1, keepdims=False)
            output = tf.matmul(concat_mean, self.neigh_weights)

        if self.use_bias:
            output += self.bias
        if self.activation:
            output = self.activation(output)

        return output

    def get_config(self):
        config = {'units': self.units,
                  'concat': self.concat,
                  'seed': self.seed
                  }

        base_config = super(MeanAggregator, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def GraphSAGE(feature_dim, neighbor_num, n_hidden, n_classes, use_bias=True, activation=tf.nn.relu,
              aggregator_type='mean', dropout_rate=0.0, l2_reg=0):
    features = Input(shape=(feature_dim,))
    node_input = Input(shape=(1,), dtype=tf.int32)
    neighbor_input = [Input(shape=(l,), dtype=tf.int32) for l in neighbor_num]
    if aggregator_type == 'mean':
        aggregator = MeanAggregator
    h = features

    for i in range(0, len(neighbor_num)):
        if i > 0:
            feature_dim = n_hidden
        if i == len(neighbor_num) - 1:
            activation = tf.nn.softmax
            n_hidden = n_classes
        h = aggregator(units=n_hidden, input_dim=feature_dim, activation=activation, l2_reg=l2_reg, use_bias=use_bias,
                       dropout_rate=dropout_rate, neigh_max=neighbor_num[i], aggregator=aggregator_type)(
            [h, node_input, neighbor_input[i]])
    output = h
    input_list = [features, node_input] + neighbor_input
    model = Model(input_list, outputs=output)
    model.__setattr__("embedding", output)

    return model
