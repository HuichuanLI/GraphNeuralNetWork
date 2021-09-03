# -*- coding:utf-8 -*-
# @Time : 2021/9/3 12:04 上午
# @Author : huichuan LI
# @File : TransD.py
# @Software: PyCharm
import tensorflow as tf


class TransD(tf.keras.Model):
    '''TransD模型类,定义TransD的参数空间和loss计算
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

        # 初始化实体语义向量空间
        self.ent_embeddings = tf.keras.layers.Embedding(
            input_dim=self.entity_total, output_dim=self.embedding_dim, name="ent_embedding",
            embeddings_initializer=tf.keras.initializers.glorot_normal(), )
        # 初始化关系翻译向量空间
        self.rel_embeddings = tf.keras.layers.Embedding(
            input_dim=self.relationship_total, output_dim=self.embedding_dim, name="rel_embedding",
            embeddings_initializer=tf.keras.initializers.glorot_normal(), )
        # 实体的转换向量
        self.ent_transfer = tf.keras.layers.Embedding(
            input_dim=self.entity_total, output_dim=self.embedding_dim, name="ent_transfer",
            embeddings_initializer=tf.keras.initializers.glorot_normal(), )
        # 关系的转换向量
        self.rel_transfer = tf.keras.layers.Embedding(
            input_dim=self.relationship_total, output_dim=self.embedding_dim, name="rel_transfer",
            embeddings_initializer=tf.keras.initializers.glorot_normal(), )

    def compute_loss(self, x):
        # 计算一个批次数据的合页损失函数值
        def _transfer(h, t, r):
            return tf.math.l2_normalize(h + tf.math.reduce_sum(h * t, 1, keepdims=True) * r, 1)

        # 获得头、尾、关系的 ID
        pos_h_id, pos_t_id, pos_r_id, neg_h_id, neg_t_id, neg_r_id = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:,
                                                                                                                  5]
        # 根据ID获得语义向量(E)和转移向量(T)
        pos_h_e = self.ent_embeddings(pos_h_id)
        pos_t_e = self.ent_embeddings(pos_t_id)
        pos_r_e = self.rel_embeddings(pos_r_id)
        pos_h_t = self.ent_transfer(pos_h_id)
        pos_t_t = self.ent_transfer(pos_t_id)
        pos_r_t = self.rel_transfer(pos_r_id)

        neg_h_e = self.ent_embeddings(neg_h_id)
        neg_t_e = self.ent_embeddings(neg_t_id)
        neg_r_e = self.rel_embeddings(neg_r_id)
        neg_h_t = self.ent_transfer(neg_h_id)
        neg_t_t = self.ent_transfer(neg_t_id)
        neg_r_t = self.rel_transfer(neg_r_id)

        pos_h_e = _transfer(pos_h_e, pos_h_t, pos_r_t)
        pos_t_e = _transfer(pos_t_e, pos_t_t, pos_r_t)
        neg_h_e = _transfer(neg_h_e, neg_h_t, neg_r_t)
        neg_t_e = _transfer(neg_t_e, neg_t_t, neg_r_t)

        if self.l1_flag:
            pos = tf.math.reduce_sum(tf.math.abs(pos_h_e + pos_r_e - pos_t_e), 1, keepdims=True)
            neg = tf.math.reduce_sum(tf.math.abs(neg_h_e + neg_r_e - neg_t_e), 1, keepdims=True)
        else:
            pos = tf.math.reduce_sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1, keepdims=True)
            neg = tf.math.reduce_sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1, keepdims=True)
        return tf.math.reduce_sum(tf.math.maximum(pos - neg + self.margin, 0))
