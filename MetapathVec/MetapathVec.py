# -*- coding:utf-8 -*-
# @Time : 2021/8/30 11:56 下午
# @Author : huichuan LI
# @File : MetapathVec.py
# @Software: PyCharm

import numpy as np

np.random.seed(12345)


class DataReader:
    NEGATIVE_TABLE_SIZE = 1e8

    def __init__(self, dataset, min_count, care_type):

        # 初始化变量
        self.negatives = []
        self.discards = []
        self.negpos = 0
        self.care_type = care_type
        self.word2id = dict()
        self.id2word = dict()
        self.sentences_count = 0
        self.token_count = 0
        self.word_frequency = dict()
        self.inputFileName = dataset

        # 执行函数，给初始化变量赋值
        # 读metapath文件
        self.read_words(min_count)
        # 产生负采样节点列表
        self.initTableNegatives()
        # word2vec中的subsampling
        self.initTableDiscards()

    def read_words(self, min_count):
        # 统计词频，将出现次数少于min_count(如5)的单词/节点过滤掉
        word_frequency = dict()
        for line in open(self.inputFileName, encoding="ISO-8859-1"):
            line = line.split()
            if len(line) > 1:
                # 统计有多少个句子/这里一个句子是一次random walk(metapath sequences)
                self.sentences_count += 1
                for word in line:
                    if len(word) > 0:
                        # 统计词数/节点数，统计重复的
                        self.token_count += 1
                        # 统计词频
                        word_frequency[word] = word_frequency.get(word, 0) + 1
                        # 输出读图过程
                        if self.token_count % 1000000 == 0:
                            print("Read " + str(int(self.token_count / 1000000)) + "M words.")

        wid = 0
        for w, c in word_frequency.items():
            # 统计词频，将出现次数少于min_count(如5)的单词/节点过滤掉
            if c < min_count:
                continue
            # word/node name -> id
            self.word2id[w] = wid
            # id -> word/node name
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            wid += 1

        # 注意留下的词全是词频 >= min_count的
        self.word_count = len(self.word2id)
        print("Total embeddings: " + str(len(self.word2id)))

    def initTableDiscards(self):
        # get a frequency table for sub-sampling. Note that the frequency is adjusted by
        # sub-sampling tricks.

        # word2vec中的subsampling
        # 最高频的词汇，比如in, the, a这些词。这样的词汇通常比其它罕见词提供了更少的信息量。
        t = 0.0001
        f = np.array(list(self.word_frequency.values())) / self.token_count
        # http://d0evi1.com/word2vec-subsampling/
        self.discards = np.sqrt(t / f) + (t / f)

    def initTableNegatives(self):
        # get a table for negative sampling, if word with index 2 appears twice, then 2 will be listed
        # in the table twice.
        # 举例: self.negatives = [1 1 1 2 2 2 2 3 3 3 4 4 4]
        pow_frequency = np.array(list(self.word_frequency.values())) ** 0.75
        words_pow = sum(pow_frequency)
        # ratio是每个词/节点负采样的概率
        ratio = pow_frequency / words_pow
        # 每个词/节点根据ratio看应该分配多少个
        count = np.round(ratio * DataReader.NEGATIVE_TABLE_SIZE)
        # 产生对应的negatives列表
        for wid, c in enumerate(count):
            self.negatives += [wid] * int(c)
        self.negatives = np.array(self.negatives)
        # 打乱顺序
        np.random.shuffle(self.negatives)
        self.sampling_prob = ratio

    def getNegatives(self, target, size):  # TODO check equality with target
        if self.care_type == 0:
            # negpos初始为0
            # 取出size个负样本
            response = self.negatives[self.negpos:self.negpos + size]
            # 挪动negpos的位置，方便下一次取negative samples
            self.negpos = (self.negpos + size) % len(self.negatives)
            # 处理negatives列表已经遍历到结尾的情况
            if len(response) != size:
                return np.concatenate((response, self.negatives[0:self.negpos]))
        return response


if __name__ == "__main__":
    min_count = 5
    care_type = 0
    data = DataReader("./net_dbis/output_path.txt", min_count, care_type)
