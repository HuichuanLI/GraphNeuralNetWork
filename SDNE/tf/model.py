import numpy as np
from tensorflow import keras
import tensorflow as tf


class MNN(keras.Model):
    def __init__(self, node_size, nhid0, nhid1, droput, alpha):
        super(MNN, self).__init__()
        self.encode0 = tf.keras.layers.Dense(node_size, nhid0)
        self.encode1 = tf.keras.layers.Dense(nhid0, nhid1)
        self.decode0 = tf.keras.layers.Dense(nhid1, nhid0)
        self.decode1 = tf.keras.layers.Dense(nhid0, node_size)
        self.droput = droput
        self.alpha = alpha

    def call(self, adj_batch, adj_mat, b_mat):
        t0 = tf.keras.layers.LeakyReLU(self.encode0(adj_batch))
        t0 = tf.keras.layers.LeakyReLU(self.encode1(t0))
        embedding = t0
        t0 = tf.keras.layers.LeakyReLU(self.decode0(t0))
        t0 = tf.keras.layers.LeakyReLU(self.decode1(t0))
        embedding_norm = tf.reduce_sum(embedding * embedding, axis=1, keepdims=True)
        L_1st = tf.reduce_sum(adj_mat * (embedding_norm -
                                         2 * tf.matmul(embedding, tf.transpose(embedding))
                                         + tf.transpose(embedding_norm)))
        L_2nd = tf.reduce_sum(((adj_batch - t0) * b_mat) * ((adj_batch - t0) * b_mat))
        return L_1st, self.alpha * L_2nd, L_1st + self.alpha * L_2nd

    def savector(self, adj):
        t0 = self.encode0(adj)
        t0 = self.encode1(t0)
        return t0
