# -*- coding:utf-8 -*-
# @Time : 2021/9/2 11:41 下午
# @Author : huichuan LI
# @File : utils.py
# @Software: PyCharm
import numpy as np
import random

import tensorflow as tf
import Config


class DataHelper():
    '''数据处理类，该类将三元组文件进行处理，得到所有的三元组数据、实体集、关系集
    '''

    def __init__(self, config):
        self.model_name = config.model_name
        self.triple_file_path = config.triple_path
        self.entity_dict = {}  # 用来存放实体和实体ID，每个元素是 实体：实体ID
        self.entity_set = set()  # 用来存放所有实体
        self.entity2entity_dict = {}  # 用来存放每个实体相连的边，每个元素是 实体：实体的集合
        self.relationship_dict = {}  # 用来存放关系和关系ID，每个元素是 关系：关系ID
        self.triple_list_list = []  # 用来存放三元组数据，每个元素是 [头实体,关系,尾实体]
        self.relationship_total = 0
        self.entity_total = 0
        self.head_set = {}  # 所有的头实体集合,每个元素是 实体：[以该实体为头实体的尾实体]
        self.tail_set = {}  # 所有的尾实体集合
        with open(self.triple_file_path) as f:
            print("Loading data from {}".format(self.triple_file_path))
            for line in f.readlines():
                h, r, t = line.strip().split(config.dividers)
                self.triple_list_list.append([h, r, t])
                if h not in self.head_set:
                    self.head_set[h] = [t]
                else:
                    self.head_set[h].append(t)
                if t not in self.tail_set:
                    self.tail_set[t] = [h]
                else:
                    self.tail_set[t].append(t)

                # 增加实体到字典
                for _ in range(2):
                    entity = [h, t][_]  # 当前头或者尾
                    other = [h, t][1 - _]  # 另一个尾或者头
                    if not entity in self.entity_dict:
                        self.entity_dict[entity] = self.entity_total
                        self.entity_total += 1
                    if not entity in self.entity2entity_dict:
                        self.entity2entity_dict[entity] = set()
                    # set会自动去重，所以每次直接添加即可
                    self.entity2entity_dict[entity].add(other)
                    self.entity_set.add(entity)
                # 增加关系到字典
                if not r in self.relationship_dict:
                    self.relationship_dict[r] = self.relationship_total
                    self.relationship_total += 1
            print("总有个三元组{}个".format(len(self.triple_list_list)))
            print("总共有实体{}个".format(self.entity_total))
            print("总共有关系{}个".format(self.relationship_total))
        total_tail_per_head = 0
        total_head_per_tail = 0
        for h in self.head_set:
            total_tail_per_head += len(self.head_set[h])
        for t in self.tail_set:
            total_head_per_tail += len(self.tail_set[t])
        self.tph = 0  # 每个头实体平均几个尾实体
        self.hpt = 0  # 每个尾实体平均几个头实体
        self.tph = total_tail_per_head / len(self.head_set)
        self.hpt = total_head_per_tail / len(self.tail_set)

    def word2id(self, word):
        """word2id的转化
        """
        if word in self.entity_dict:
            result = self.entity_dict[word]
        elif word in self.relationship_dict:
            result = self.relationship_dict[word]
        else:
            exit(1)
        return result

    def get_negative_entity(self, entity):
        """替换entity,获得不存在的三元组
        """
        return np.random.choice(list(self.entity_set - self.entity2entity_dict[entity]))

    def get_tf_dataset(self):
        """获得训练集，验证集，测试集
        格式为:[pos_h_id,pos_t_id,pos_r_id,neg_h_id,neg_t_id,neg_r_id]
        """
        data_list = []
        print("Creating data")
        for triple_list in self.triple_list_list:
            # 每个存在的三元组要对应两个不存在的三元组，参见原文
            temp_list1 = [
                triple_list[0], triple_list[2], triple_list[1],
                self.get_negative_entity(triple_list[0]), triple_list[2], triple_list[1]
            ]
            temp_list2 = [
                triple_list[0], triple_list[2], triple_list[1],
                triple_list[0], self.get_negative_entity(triple_list[2]), triple_list[1]
            ]
            if self.model_name == 'transd':
                data_list.extend([[self.word2id(v) for v in temp_list1], [self.word2id(v) for v in temp_list2]])
            elif self.model_name in ['transh', 'transr']:
                if random.random() < (self.tph / (self.tph + self.hpt)):
                    data_list.extend([[self.word2id(v) for v in temp_list1]])
                else:
                    data_list.extend([[self.word2id(v) for v in temp_list2]])
            else:  # transe or default
                if random.random() < 0.5:  # 随机对头结点或尾结点进行伪造
                    data_list.extend([[self.word2id(v) for v in temp_list1]])
                else:
                    data_list.extend([[self.word2id(v) for v in temp_list2]])
        print(np.array(data_list).shape)
        print("Created data")
        return tf.data.Dataset.from_tensor_slices(data_list)


if __name__ == "__main__":
    cf = Config.Config()
    datahelper = DataHelper(config=cf)
