# -*- coding:utf-8 -*-
# @Time : 2021/8/28 5:22 下午
# @Author : huichuan LI
# @File : Line.py
# @Software: PyCharm
from tensorflow import keras
import tensorflow as tf


class Line(keras.Model):
    def __init__(self, size, embed_dim=128, order=1):
        super(Line, self).__init__()

        assert order in [1, 2], print("Order should either be int(1) or int(2)")

        self.embed_dim = embed_dim
        self.order = order
        self.nodes_embeddings = keras.layers.Embedding(size, embed_dim,
                                                       embeddings_initializer=keras.initializers.RandomNormal(0., 0.1),
                                                       )

        if order == 2:
            self.contextnodes_embeddings = keras.layers.Embedding(size, embed_dim,
                                                                  embeddings_initializer=keras.initializers.RandomNormal(
                                                                      0., 0.1),
                                                                  )

    def call(self, v_i, v_j, negsamples, device):

        v_i = self.nodes_embeddings(v_i)

        if self.order == 2:
            v_j = self.contextnodes_embeddings(v_j)
            negativenodes = -self.contextnodes_embeddings(negsamples)

        else:
            v_j = self.nodes_embeddings(v_j)
            negativenodes = -self.nodes_embeddings(negsamples)

        mulpositivebatch = tf.multiply(v_i, v_j)
        positivebatch = tf.keras.activations.sigmoid(tf.reduce_sum(mulpositivebatch, axis=1))

        mulnegativebatch = tf.multiply(
            tf.reshape(len(v_i), 1, self.embed_dim), negativenodes)
        negativebatch = tf.reduce_sum(
            tf.keras.activations.sigmoid(
                tf.reduce_sum(mulnegativebatch, dim=2)
            ),
            axis=1)
        loss = positivebatch + negativebatch
        return -tf.reduce_mean(loss)
