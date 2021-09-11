# -*- coding:utf-8 -*-
# @Time : 2021/9/11 10:32 下午
# @Author : huichuan LI
# @File : main.py
# @Software: PyCharm

import scipy.sparse as sp
import numpy as np
from gcn import GCN
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from utils import load_data
import tensorflow

if __name__ == "__main__":
    # Read data

    FEATURE_LESS = False

    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(path="../Graph/data/cora/")

    if FEATURE_LESS:
        X = np.arange(adj.shape[-1])
        feature_dim = adj.shape[-1]
    else:
        X = features
        feature_dim = X.shape[-1]

    model_input = [X.toarray(), adj.toarray()]

    model = GCN(adj.shape[-1], feature_dim, 16, y_train.shape[1], dropout_rate=0.5, l2_reg=2.5e-4,
                feature_less=FEATURE_LESS, )

    model.compile(optimizer=Adam(0.01), loss='categorical_crossentropy',
                  weighted_metrics=['categorical_crossentropy', 'acc', tensorflow.keras.metrics.AUC(name='auc'),
                                    ])

    NB_EPOCH = 200
    PATIENCE = 200  # early stopping patience

    val_data = (model_input, y_val, val_mask)
    # train
    print("start training")
    model.fit(model_input, y_train, sample_weight=train_mask, validation_data=val_data,
              batch_size=adj.shape[0], epochs=NB_EPOCH, shuffle=False, verbose=2)

    model_input = [adj.toarray(), features.toarray()]
    user_embedding_model = Model(inputs=[model.adj_input, model.feature_input], outputs=model.embedding)

    user_embs = user_embedding_model.predict(model_input, batch_size=adj.shape[0])
    print(user_embs.shape)
