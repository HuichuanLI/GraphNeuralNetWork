# -*- coding:utf-8 -*-
# @Time : 2021/9/2 11:48 下午
# @Author : huichuan LI
# @File : TransH.py
# @Software: PyCharm

import tensorflow as tf


class TransH(tf.keras.Model):
    '''TransH模型类，定义了TransH的参数空间和loss计算
    '''

    def __init__(self, config, data_helper):
        super().__init__()
        self.entity_total = data_helper.entity_total  # 实体总数
        self.relationship_total = data_helper.relationship_total  # 关系总数
        self.l1_flag = config.l1_flag  # L1正则化
        self.margin = config.margin  # 合页损失函数中的样本差异度值
        self.entity_embeddings_file_path = config.entity_embeddings_path  # 存储实体embeddings的文件
        self.relationship_embeddings_file_path = config.relationship_embeddings_path  # 存储关系embeddings的文件
        self.embedding_dim = config.embedding_dim  # 向量维度
        self.epsilon = config.epsilon  # 软约束中的对于法向量和翻译向量的超参
        self.C = config.C  # 软约束的参数
        # 初始化实体语义向量空间
        self.ent_embeddings = tf.keras.layers.Embedding(
            input_dim=self.entity_total, output_dim=self.embedding_dim, name="ent_embedding",
            embeddings_initializer=tf.keras.initializers.glorot_normal(), )
        # 初始化关系翻译向量空间
        self.rel_embeddings = tf.keras.layers.Embedding(
            input_dim=self.relationship_total, output_dim=self.embedding_dim, name="rel_embedding",
            embeddings_initializer=tf.keras.initializers.glorot_normal(), )
        # 初始化关系超平面法向量空间
        self.norm_embeddings = tf.keras.layers.Embedding(
            input_dim=self.relationship_total, output_dim=self.embedding_dim, name="rel_embedding",
            embeddings_initializer=tf.keras.initializers.glorot_normal(), )

    def compute_loss(self, x):
        # 计算一个批次数据的合页损失函数值
        # 获得头、尾、关系的 ID
        def _transfer(e, norm):
            # 在关系超平面上做映射，得到在关系超平面上的映射向量
            norm = tf.norm(norm, ord=2, axis=1)  # 模长为1的法向量
            return e - tf.math.reduce_sum(e * norm, 1, keepdims=True) * norm

        pos_h_id, pos_t_id, pos_r_id, neg_h_id, neg_t_id, neg_r_id = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:,
                                                                                                                  5]
        pos_h_e = self.ent_embeddings(pos_h_id)
        pos_t_e = self.ent_embeddings(pos_t_id)
        pos_r_e = self.rel_embeddings(pos_r_id)
        pos_r_n = self.norm_embeddings(pos_r_id)  # 正例关系法向量

        neg_h_e = self.ent_embeddings(neg_h_id)
        neg_t_e = self.ent_embeddings(neg_t_id)
        neg_r_e = self.rel_embeddings(neg_r_id)
        neg_r_n = self.norm_embeddings(neg_r_id)  # 负例关系法向量

        # 获取到当前的h上面
        pos_h_e = _transfer(pos_h_e, pos_r_n)
        pos_t_e = _transfer(pos_t_e, pos_r_n)
        neg_h_e = _transfer(neg_h_e, neg_r_n)
        neg_t_e = _transfer(neg_t_e, neg_r_n)

        if self.l1_flag:
            pos = tf.math.reduce_sum(tf.math.abs(pos_h_e + pos_r_e - pos_t_e), 1, keepdims=True)
            neg = tf.math.reduce_sum(tf.math.abs(neg_h_e + neg_r_e - neg_t_e), 1, keepdims=True)
        else:
            pos = tf.math.reduce_sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1, keepdims=True)
            neg = tf.math.reduce_sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1, keepdims=True)
        hinge_loss = tf.math.reduce_sum(tf.math.maximum(pos - neg + self.margin, 0))  # 合页损失
        entity_loss = 0
        for e in self.ent_embeddings:
            embedding = self.ent_embeddings[e]
            entity_loss += tf.math.maximum(0, tf.math.reduce_sum(tf.norm(embedding, ord=2, axis=1)) - 1)
        relationship_loss = 0
        for r in self.rel_embeddings:
            w = self.norm_embeddings[r]  # 法向量
            d = self.rel_embeddings[r]  # 翻译向量
            relationship_loss += tf.math.maximum(0,
                                                 (tf.math.reduce_sum(tf.matmul(tf.transpose(w), d)) / \
                                                  tf.math.reduce_sum(tf.norm(d, ord=2, axis=1))) - self.epsilon ** 2)
        loss = 0
        loss += hinge_loss
        loss += self.C(entity_loss + relationship_loss)
        return loss
