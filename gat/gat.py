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

    def __init__(self, att_embedding_size=8, head_num=8, dropout_rate=0.5, l2_reg=0, activation=tf.nn.relu,
                 reduction='concat', use_bias=True, seed=1024, **kwargs):
        if head_num <= 0:
            raise ValueError('head_num must be a int > 0')
        self.att_embedding_size = att_embedding_size
        self.head_num = head_num
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.activation = activation
        self.act = activation
        self.reduction = reduction
        self.use_bias = use_bias
        self.seed = seed
        super(GATLayer, self).__init__(**kwargs)
