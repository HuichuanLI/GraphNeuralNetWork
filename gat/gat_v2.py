# -*- coding:utf-8 -*-
# @Time : 2021/9/5 4:45 下午
# @Author : huichuan LI
# @File : gat.py
# @Software: PyCharm
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import Zeros
from tensorflow.python.keras.layers import Layer, Dropout, Input
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.models import Model


class GATLayer(Layer):

    def __init__(self, in_features, out_features, adj_dim, dropout_rate=0.5, l2_reg=0, activation=tf.nn.relu,
                 reduction='concat', seed=1024, **kwargs):

        self.in_features = in_features
        self.out_features = out_features
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.activation = activation
        self.act = activation
        self.reduction = reduction
        self.seed = seed
        self.adj_dim = adj_dim
        super(GATLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        X, A = input_shape
        embedding_size = int(X[-1])
        self.weight = self.add_weight(name='weight', shape=[self.in_features, self.out_features],
                                      dtype=tf.float32,
                                      regularizer=l2(self.l2_reg),
                                      initializer=tf.keras.initializers.glorot_uniform())
        self.a = self.add_weight(name='att_self_weight',
                                 shape=[2 * self.out_features, 1],
                                 dtype=tf.float32,
                                 regularizer=l2(self.l2_reg),
                                 initializer=tf.keras.initializers.glorot_uniform())

        # if self.use_bias:
        #     self.bias_weight = self.add_weight(name='bias', shape=[1, self.head_num, self.att_embedding_size],
        #                                        dtype=tf.float32,
        #                                        initializer=Zeros())
        self.in_dropout = Dropout(self.dropout_rate)
        self.feat_dropout = Dropout(self.dropout_rate, )
        self.att_dropout = Dropout(self.dropout_rate, )
        # Be sure to call this somewhere!
        super(GATLayer, self).build(input_shape)

    def _prepare_attentional_mechanism_input(self, wh):
        N = self.adj_dim  # number of nodes
        Wh_repeated_in_chunks = tf.repeat(wh, repeats=N, axis=0)
        Wh_repeated_alternating = tf.tile(wh, [N, 1])
        all_combinations_matrix = tf.concat([Wh_repeated_in_chunks, Wh_repeated_alternating], axis=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)
        all_combinations_matrix = tf.reshape(all_combinations_matrix, [N, N, 2 * self.out_features])
        return all_combinations_matrix

    def call(self, inputs, training=None, **kwargs):
        X, A = inputs
        X = self.in_dropout(X)  # N * D
        if K.ndim(X) != 2:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 2 dimensions" % (K.ndim(X)))

        features = tf.matmul(X, self.weight, )  # None * output
        Wh = features
        a_input = self._prepare_attentional_mechanism_input(wh=Wh)
        # 即论文里的eij
        # squeeze除去维数为1的维度
        # [2708, 2708, 16]与[16, 1]相乘再除去维数为1的维度，故其维度为[2708,2708],与领接矩阵adj的维度一样

        e = tf.squeeze(self.act(tf.matmul(a_input, self.a)), axis=2)

        # mask-attention

        # 维度大小与e相同，所有元素都是-9*10的15次方
        zero_vec = -9e15 * tf.ones_like(e)

        # 故adj的领接矩阵的大小为[2708, 2708] (归一化处理之后的)
        # 故当adj>0，即两结点有边，则用gat构建的矩阵e，若adj=0,则另其为一个很大的负数，这么做的原因是进行softmax时，这些数就会接近于0了
        attention = tf.where(A > 0, e, zero_vec)

        # 对应论文公式3，attention就是公式里的a_ij
        attention = tf.nn.softmax(attention, axis=1)
        attention = self.att_dropout(attention)
        result = tf.matmul(attention, Wh)

        # head_num Node embeding_size
        if self.reduction == "concat":
            result = tf.nn.elu(result)
            return result
        else:
            return result

    def compute_output_shape(self, input_shape):
        if self.reduction == "concat":

            return (None, self.att_embedding_size * self.head_num)
        else:
            return (None, self.att_embedding_size)

    def get_config(self, ):
        config = {'att_embedding_size': self.att_embedding_size, 'head_num': self.head_num, 'use_res': self.use_res,
                  'seed': self.seed}
        base_config = super(GATLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def GAT(adj_dim, feature_dim, num_class, n_attn_heads=8, att_embedding_size=8, dropout_rate=0.0,
        l2_reg=0.0, use_bias=True):
    X_in = Input(shape=(feature_dim,))
    A_in = Input(shape=(adj_dim,))
    h = X_in

    attentions = tf.concat(
        [GATLayer(in_features=feature_dim, out_features=att_embedding_size, dropout_rate=dropout_rate,
                  l2_reg=l2_reg, adj_dim=adj_dim,
                  activation=tf.nn.elu)([h, A_in]) for _ in
         range(n_attn_heads)], axis=1)
    attentions = tf.nn.elu(attentions)

    h = GATLayer(in_features=att_embedding_size * n_attn_heads, adj_dim=adj_dim, out_features=num_class,
                 dropout_rate=dropout_rate,
                 l2_reg=l2_reg,
                 activation=tf.nn.softmax, reduction='mean')([attentions, A_in])
    model = Model(inputs=[X_in, A_in], outputs=h)

    return model
