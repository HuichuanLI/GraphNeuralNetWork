from utils import osUtils as ou
from data_set import filepaths as fp
from torch.utils.data import Dataset, DataLoader
import sys
import random
import copy


def readKGData(path=fp.Ml_100K.KG):
    print('读取知识图谱数据...')
    entity_set = set()
    relation_set = set()
    triples = []
    for h, r, t in ou.readTriple(path):
        entity_set.add(int(h))
        entity_set.add(int(t))
        relation_set.add(int(r))
        triples.append([int(h), int(r), int(t)])
    return list(entity_set), list(relation_set), triples


def readRecData(path=fp.Ml_100K.RATING, test_ratio=0.1):
    print('读取用户评分三元组...')
    user_set, item_set = set(), set()
    triples = []
    for u, i, r in ou.readTriple(path):
        user_set.add(int(u))
        item_set.add(int(i))
        triples.append((int(u), int(i), int(r)))

    test_set = random.sample(triples, int(len(triples) * test_ratio))
    train_set = list(set(triples) - set(test_set))
    # 返回用户集合列表，物品集合列表，与用户，物品，评分三元组列表
    return list(user_set), list(item_set), train_set, test_set


from torch.utils.data import Dataset


# 继承torch自带的Dataset类,重构__getitem__与__len__方法
class KgDatasetWithNegativeSampling(Dataset):

    def __init__(self, triples, entitys):
        self.triples = triples  # 知识图谱HRT三元组
        self.entitys = entitys  # 所有实体集合列表

    def __getitem__(self, index):
        '''
        :param index: 一批次采样的列表索引序号
        '''
        # 根据索引取出正例
        pos_triple = self.triples[index]
        # 通过负例采样的方法得到负例
        neg_triple = self.negtiveSampling(pos_triple)
        return pos_triple, neg_triple

    # 负例采样方法
    def negtiveSampling(self, triple):
        seed = random.random()
        neg_triple = copy.deepcopy(triple)
        if seed > 0.5:  # 替换head
            rand_head = triple[0]
            while rand_head == triple[0]:  # 如果采样得到自己则继续循环
                # 从所有实体中随机采样一个实体
                rand_head = random.sample(self.entitys, 1)[0]
            neg_triple[0] = rand_head
        else:  # 替换tail
            rand_tail = triple[2]
            while rand_tail == triple[2]:
                rand_tail = random.sample(self.entitys, 1)[0]
            neg_triple[2] = rand_tail
        return neg_triple

    def __len__(self):
        return len(self.triples)


if __name__ == '__main__':
    # 读取文件得到所有实体列表，所有关系列表，以及HRT三元组列表
    entitys, relations, triples = readKGData()
    # 传入HRT三元与所有实体得到包含正例与负例三元组的data set
    train_set = KgDatasetWithNegativeSampling(triples, entitys)
    from torch.utils.data import DataLoader

    # 通过torch的 DataLoader方法批次迭代三元组数据
    for set in DataLoader(train_set, batch_size=8, shuffle=True):
        # 将正负例数据拆解开
        pos_set, neg_set = set
        # 可以打印一下看看
        print(pos_set)
        print(neg_set)
        sys.exit()
