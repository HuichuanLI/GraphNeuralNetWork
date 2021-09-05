# -*- coding:utf-8 -*-
# @Time : 2021/9/5 6:11 下午
# @Author : huichuan LI
# @File : main.py
# @Software: PyCharm

import scipy.sparse as sp
import numpy as np
from gat_v2 import GAT
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from utils import load_data

if __name__ == "__main__":
    # Read data

    FEATURE_LESS = False

    adj, features, labels, idx_train, idx_val, idx_test = load_data(path="../Graph/data/cora/")

    print(labels.shape[1])
    model = GAT(adj_dim=adj.shape[0], feature_dim=features.shape[1], num_class=labels.shape[1],
                n_attn_heads=8, att_embedding_size=8,
                dropout_rate=0.6, l2_reg=2.5e-4)

    optimizer = Adam(lr=0.005)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  weighted_metrics=['categorical_crossentropy', 'acc'])
