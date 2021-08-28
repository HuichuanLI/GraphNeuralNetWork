# -*- coding:utf-8 -*-
# @Time : 2021/8/28 5:22 下午
# @Author : huichuan LI
# @File : Line.py
# @Software: PyCharm
from tensorflow import keras
import tensorflow as tf


class Line(keras.Model):
    def __init__(self, size, embed_dim=128, order=1):
        super(Line, self).__init__()

        assert order in [1, 2], print("Order should either be int(1) or int(2)")

        self.embed_dim = embed_dim
        self.order = order
        self.nodes_embeddings = keras.layers.Embedding(size, embed_dim,
                                                       embeddings_initializer=keras.initializers.RandomNormal(0., 0.1),
                                                       )

        if order == 2:
            self.contextnodes_embeddings = keras.layers.Embedding(size, embed_dim,
                                                                  embeddings_initializer=keras.initializers.RandomNormal(
                                                                      0., 0.1),
                                                                  )

    def call(self, v_i, v_j, negsamples):

        v_i = self.nodes_embeddings(v_i)

        if self.order == 2:
            v_j = self.contextnodes_embeddings(v_j)
            negativenodes = -self.contextnodes_embeddings(negsamples)

        else:
            v_j = self.nodes_embeddings(v_j)
            negativenodes = -self.nodes_embeddings(negsamples)

        mulpositivebatch = tf.multiply(v_i, v_j)
        positivebatch = tf.keras.activations.sigmoid(tf.reduce_sum(mulpositivebatch, axis=1))

        mulnegativebatch = tf.multiply(
            tf.reshape(v_i, [len(v_i), 1, self.embed_dim]), negativenodes)
        negativebatch = tf.reduce_sum(
            tf.keras.activations.sigmoid(
                tf.reduce_sum(mulnegativebatch, axis=2)
            ),
            axis=1)
        loss = positivebatch + negativebatch
        return -tf.reduce_mean(loss)

    def train(self, v_i, v_j, negsamples):
        with tf.GradientTape() as tape:
            loss = self.call(self, v_i, v_j, negsamples)
        gradients = tape.gradient(loss, self.policygrad_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.policygrad_model.trainable_variables))

        return loss.numpy()


import random
from decimal import *
import numpy as np
import collections
from tqdm import tqdm


class VoseAlias(object):
    """
    构建alias table,达到O(1)的采样效率
    Adding a few modifs to https://github.com/asmith26/Vose-Alias-Method
    """

    def __init__(self, dist):
        """
        初始化函数
        (VoseAlias, dict) -> NoneType
        """
        self.dist = dist
        self.alias_initialisation()

    def alias_initialisation(self):
        """
        Construct probability and alias tables for the distribution.
        """
        # Initialise variables
        n = len(self.dist)
        # 概率表
        self.table_prob = {}  # probability table
        # 替身表
        self.table_alias = {}  # alias table
        # 乘以n的概率表
        scaled_prob = {}  # scaled probabilities
        # 存储概率值小于1的
        small = []  # stack for probabilities smaller that 1
        # 存储概率值大于1的
        large = []  # stack for probabilities greater than or equal to 1

        # Construct and sort the scaled probabilities into their appropriate stacks
        # 将各个概率分成两组，一组的概率值大于1，另一组的概率值小于1
        print("1/2. Building and sorting scaled probabilities for alias table...")
        for o, p in tqdm(self.dist.items()):
            scaled_prob[o] = Decimal(p) * n

            if scaled_prob[o] < 1:
                small.append(o)
            else:
                large.append(o)

        print("2/2. Building alias table...")
        # Construct the probability and alias tables
        # 使用贪心算法，将概率值小于1的不断填满
        while small and large:
            s = small.pop()
            l = large.pop()

            self.table_prob[s] = scaled_prob[s]
            self.table_alias[s] = l
            # 更新概率值
            scaled_prob[l] = (scaled_prob[l] + scaled_prob[s]) - Decimal(1)

            if scaled_prob[l] < 1:
                small.append(l)
            else:
                large.append(l)

        # The remaining outcomes (of one stack) must have probability 1
        # 当两方不全有元素时，仅有一方有元素的也全为1
        while large:
            self.table_prob[large.pop()] = Decimal(1)

        while small:
            self.table_prob[small.pop()] = Decimal(1)
        self.listprobs = list(self.table_prob)

    def alias_generation(self):
        """
        Yields a random outcome from the distribution.
        """
        # Determine which column of table_prob to inspect
        col = random.choice(self.listprobs)
        # Determine which outcome to pick in that column
        # 取自己
        if self.table_prob[col] >= random.uniform(0, 1):
            return col
        # 取替身，即取alias table存的节点
        else:
            return self.table_alias[col]

    def sample_n(self, size):
        """
        调用alias_generation一共n次，采样n个nodes
        Yields a sample of size n from the distribution, and print the results to stdout.
        """
        for i in range(size):
            yield self.alias_generation()


def negSampleBatch(sourcenode, targetnode, negsamplesize, weights,
                   nodedegrees, nodesaliassampler, t=10e-3):
    """
    For generating negative samples.
    """
    negsamples = 0
    while negsamples < negsamplesize:
        # nodesaliassampler是实现alias building的VoseAlias类，这里采样点
        samplednode = nodesaliassampler.sample_n(1)
        # 如果采样出source或target均跳过
        if (samplednode == sourcenode) or (samplednode == targetnode):
            continue
        # 输出负样本点，一共negsamplesize个点
        else:
            negsamples += 1
            yield samplednode


def makeData(samplededges, negsamplesize, weights, nodedegrees, nodesaliassampler):
    for e in samplededges:
        sourcenode, targetnode = e[0], e[1]
        negnodes = []
        # 采样出negsamplesize个负样本点
        for negsample in negSampleBatch(sourcenode, targetnode, negsamplesize,
                                        weights, nodedegrees, nodesaliassampler):
            for node in negsample:
                negnodes.append(node)
        # 格式是(node i, node j, negative nodes...)
        yield [e[0], e[1]] + negnodes


negativepower = 0.75

edgedistdict, nodedistdict, weights, nodedegrees, maxindex = makeDist(
    '../../Graph/weighted.karate.edgelist', negativepower)

# 构建alias table,达到O(1)的采样效率
edgesaliassampler = VoseAlias(edgedistdict)
nodesaliassampler = VoseAlias(nodedistdict)

# 按batchsize将训练样本分组

opt = keras.optimizers.Adam(0.01)

batchrange = int(len(edgedistdict) / 5)
print(maxindex)
# line.py中的nn.Module类
line = Line(maxindex + 1, embed_dim=128, order=2)
# # # SGD算法优化模型
# opt = optim.SGD(line.parameters(), lr=args.learning_rate,
#             momentum=0.9, nestero|v=True)


lossdata = {"it": [], "loss": []}
it = 0
helper = 0

# 共训练epoch次数
for epoch in range(20):
    print("Epoch {}".format(epoch))
    # 每次训练组数：batchsize
    for b in range(batchrange):
        # edgesaliassampler是实现alias building的VoseAlias类，这里采样出batchsize条边
        samplededges = edgesaliassampler.sample_n(5)
        # makeData是utils.py中的函数，为每条边采样出K条负样本边
        # 每一条格式是(node i, node j, negative nodes...)
        batch = list(makeData(samplededges, 5, weights, nodedegrees,
                              nodesaliassampler))
        # 转换成tensor格式
        batch = tf.convert_to_tensor(batch)
        if helper == 0:
            print(batch)
            helper = 1
        # 第0列
        v_i = batch[:, 0]
        # 第1列
        v_j = batch[:, 1]
        # 第2列-最后列

        negsamples = batch[:, 2:]
        # 在做BP之前将gradients置0因为是累加的
        # Line模型实现部分
        with tf.GradientTape() as tape:
            loss = line(v_i, v_j, negsamples)
        gradients = tape.gradient(loss, line.trainable_variables)
        opt.apply_gradients(zip(gradients, line.trainable_variables))
        # 计算梯度
        # 根据梯度值更新参数值
        print(loss.numpy())

        lossdata["loss"].append(loss.numpy())
        lossdata["it"].append(it)
        it += 1
