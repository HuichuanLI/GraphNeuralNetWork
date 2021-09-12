# -*- coding:utf-8 -*-
# @Time : 2021/9/12 12:26 上午
# @Author : huichuan LI
# @File : eges.py
# @Software: PyCharm
import numpy as np
import tensorflow as tf
from tensorflow import keras


class EGES_Model(keras.Model):
    def __init__(self, num_nodes, num_feat, feature_lens, n_sampled=100, embedding_dim=128, lr=0.001, **kwargs):
        self.n_samped = n_sampled
        self.num_feat = num_feat
        self.feature_lens = feature_lens
        self.embedding_dim = embedding_dim
        self.num_nodes = num_nodes
        self.lr = lr
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        super(EGES_Model, self).__init__(**kwargs)

    def build(self, input_shapes):
        # noise-contrastive estimation
        self.nce_w = self.add_weight(
            name="nce_w", shape=[self.num_nodes, self.embedding_dim],
            initializer=keras.initializers.TruncatedNormal(0., 0.1))  # [n_vocab, emb_dim]
        self.nce_b = self.add_weight(
            name="nce_b", shape=(self.num_nodes,),
            initializer=keras.initializers.Constant(0.1))  # [n_vocab, ]

        cat_embedding_vars = []
        for i in range(self.num_feat):
            embedding_var = self.add_weight(
                shape=[self.feature_lens[i], self.embedding_dim]
                , initializer=keras.initializers.TruncatedNormal(0., 0.1),
                name='embedding' + str(i),
                trainable=True)
            cat_embedding_vars.append(embedding_var)
        self.cat_embedding = cat_embedding_vars
        self.alpha_embedding = self.add_weight(
            name="nce_b", shape=(self.num_nodes, self.num_feat),
            initializer=keras.initializers.Constant(0.1))

    def attention_merge(self):
        embed_list = []
        for i in range(self.num_feat):
            cat_embed = tf.nn.embedding_lookup(self.cat_embedding[i], self.batch_features[:, i])
            embed_list.append(cat_embed)
        stack_embed = tf.stack(embed_list, axis=-1)
        # attention merge
        alpha_embed = tf.nn.embedding_lookup(self.alpha_embedding, self.batch_features[:, 0])
        alpha_embed_expand = tf.expand_dims(alpha_embed, 1)
        alpha_i_sum = tf.reduce_sum(tf.exp(alpha_embed_expand), axis=-1)
        merge_emb = tf.reduce_sum(stack_embed * tf.exp(alpha_embed_expand), axis=-1) / alpha_i_sum
        return merge_emb

    def make_skipgram_loss(self, labels):
        loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(
            weights=self.nce_w,
            biases=self.nce_b,
            labels=tf.expand_dims(labels, axis=1),
            inputs=self.merge_emb,
            num_sampled=self.n_samped,
            num_classes=self.num_nodes))

        return loss

    def call(self, side_info, batch_index, batch_labels):
        self.side_info = tf.convert_to_tensor(side_info)
        self.batch_features = tf.nn.embedding_lookup(self.side_info, batch_index)

        embed_list = []
        for i in range(self.num_feat):
            cat_embed = tf.nn.embedding_lookup(self.cat_embedding[i], self.batch_features[:, i])
            embed_list.append(cat_embed)
        stack_embed = tf.stack(embed_list, axis=-1)
        # attention merge
        alpha_embed = tf.nn.embedding_lookup(self.alpha_embedding, self.batch_features[:, 0])
        alpha_embed_expand = tf.expand_dims(alpha_embed, 1)
        alpha_i_sum = tf.reduce_sum(tf.exp(alpha_embed_expand), axis=-1)
        self.merge_emb = tf.reduce_sum(stack_embed * tf.exp(alpha_embed_expand), axis=-1) / alpha_i_sum

        return self.make_skipgram_loss(batch_labels)

    def get_embedding(self, batch_index):
        self.batch_features = tf.nn.embedding_lookup(self.side_info, batch_index)

        embed_list = []
        for i in range(self.num_feat):
            cat_embed = tf.nn.embedding_lookup(self.cat_embedding[i], self.batch_features[:, i])
            embed_list.append(cat_embed)
        stack_embed = tf.stack(embed_list, axis=-1)
        # attention merge
        alpha_embed = tf.nn.embedding_lookup(self.alpha_embedding, self.batch_features[:, 0])
        alpha_embed_expand = tf.expand_dims(alpha_embed, 1)
        alpha_i_sum = tf.reduce_sum(tf.exp(alpha_embed_expand), axis=-1)
        merge_emb = tf.reduce_sum(stack_embed * tf.exp(alpha_embed_expand), axis=-1) / alpha_i_sum
        return merge_emb
