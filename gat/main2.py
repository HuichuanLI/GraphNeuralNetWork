# -*- coding:utf-8 -*-
# @Time : 2021/9/5 6:11 下午
# @Author : huichuan LI
# @File : main.py
# @Software: PyCharm

import scipy.sparse as sp
import numpy as np
from gat import GAT
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from utils import load_data

if __name__ == "__main__":
    # Read data

    FEATURE_LESS = False

    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(path="../Graph/data/cora/")

    model = GAT(adj_dim=adj.shape[0], feature_dim=features.shape[1], num_class=y_train.shape[1], num_layers=2,
                n_attn_heads=8, att_embedding_size=8,
                dropout_rate=0.6, l2_reg=2.5e-4, use_bias=True)

    optimizer = Adam(lr=0.005)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  weighted_metrics=['categorical_crossentropy', 'acc'])
    features.sort_indices()
    model_input = [features.toarray(), adj.toarray()]
    val_data = (model_input, y_val, val_mask)

    print("start training")

    model.fit(model_input, y_train, sample_weight=train_mask, validation_data=val_data,
              batch_size=adj.shape[0], epochs=200, shuffle=False, verbose=2)
