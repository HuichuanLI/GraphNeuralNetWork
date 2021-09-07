# -*- coding:utf-8 -*-
# @Time : 2021/9/7 10:45 下午
# @Author : huichuan LI
# @File : main.py
# @Software: PyCharm
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras.models import Model
from utils import load_data, sample_neighs


if __name__ == "__main__":
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(path="../Graph/data/cora/")
    indexs = np.arange(adj.shape[0])
    neigh_number = [10, 25]
    neigh_maxlen = []

    model_input = [features, np.asarray(indexs, dtype=np.int32)]
    for num in neigh_number:
        sample_neigh, sample_neigh_len = sample_neighs(
            adj, indexs, num, self_loop=False)
        model_input.extend([sample_neigh])
        neigh_maxlen.append(max(sample_neigh_len))
    model = GraphSAGE(feature_dim=features.shape[1],
                      neighbor_num=neigh_maxlen,
                      n_hidden=16,
                      n_classes=y_train.shape[1],
                      use_bias=True,
                      activation=tf.nn.relu,
                      aggregator_type='mean',
                      dropout_rate=0.5, l2_reg=2.5e-4)
    model.compile(Adam(0.01), 'categorical_crossentropy',
                  weighted_metrics=['categorical_crossentropy', 'acc'])
