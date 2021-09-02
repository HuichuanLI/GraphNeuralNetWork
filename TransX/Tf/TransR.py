# -*- coding:utf-8 -*-
# @Time : 2021/9/2 11:59 下午
# @Author : huichuan LI
# @File : TransR.py
# @Software: PyCharm
import tensorflow as tf


class TrasnR(tf.keras.Model):
    '''TransR模型类,定义TransR的参数空间和loss计算
    '''

    def __init__(self, config, data_helper):
        super().__init__()
        self.entity_total = data_helper.entity_total  # 实体总数
        self.relationship_total = data_helper.relationship_total  # 关系总数
        self.l1_flag = config.l1_flag  # L1正则化
        self.margin = config.margin  # 合页损失函数中的样本差异度值
        self.entity_embeddings_file_path = config.entity_embeddings_path  # 存储实体embeddings的文件
        self.relationship_embeddings_file_path = config.relationship_embeddings_path  # 存储关系embeddings的文件
        self.entity_embedding_dim = config.entity_embedding_dim  # 实体向量维度
        self.rel_embedding_dim = config.rel_embedding_dim  # 关系向量维度
        # 初始化实体语义向量空间
        self.ent_embeddings = tf.keras.layers.Embedding(
            input_dim=self.entity_total, output_dim=self.entity_embedding_dim, name="ent_embedding",
            embeddings_initializer=tf.keras.initializers.glorot_normal(), )
        # 初始化关系向量空间
        self.rel_embeddings = tf.keras.layers.Embedding(
            input_dim=self.relationship_total, output_dim=self.rel_embedding_dim
            , name="rel_embedding", embeddings_initializer=tf.keras.initializers.glorot_normal(), )
        # 初始化实体-关系翻译矩阵空间
        self.transfer_matrix = tf.keras.layers.Embedding(
            input_dim=self.relationship_total, output_dim=self.entity_embedding_dim * self.rel_embedding_dim
            , name="rel_embedding", embeddings_initializer=tf.keras.initializers.glorot_normal(), )

    def compute_loss(self, x, L1_flag):
        def _transfer(e, r_id):
            matrix = self.transfer_matrix(r_id).reapse(self.entity_embedding_dim, self.rel_embedding_dim)
            return tf.math.l2_normalize(e * matrix, 1)

        pos_h_id, pos_t_id, pos_r_id, neg_h_id, neg_t_id, neg_r_id = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:,
                                                                                                                  5]
        # 根据ID获得语义向量(E)和转移向量(T)
        pos_h_e = self.ent_embeddings(pos_h_id)
        pos_t_e = self.ent_embeddings(pos_t_id)
        pos_r_e = self.rel_embeddings(pos_r_id)

        neg_h_e = self.ent_embeddings(neg_h_id)
        neg_t_e = self.ent_embeddings(neg_t_id)
        neg_r_e = self.rel_embeddings(neg_r_id)

        pos_h_e = _transfer(pos_h_e, pos_r_id)
        pos_t_e = _transfer(pos_t_e, pos_r_id)
        neg_h_e = _transfer(neg_h_e, neg_r_id)
        neg_t_e = _transfer(neg_t_e, neg_r_id)

        if L1_flag:
            pos = tf.reduce_sum(abs(pos_h_e + pos_r_e - pos_t_e), 1, keepdims=True)
            neg = tf.reduce_sum(abs(neg_h_e + neg_r_e - neg_t_e), 1, keepdims=True)
            self.predict = pos
        else:
            pos = tf.reduce_sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1, keepdims=True)
            neg = tf.reduce_sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1, keepdims=True)
            self.predict = pos
        return tf.reduce_sum(tf.maximum(pos - neg + self.margin, 0))
