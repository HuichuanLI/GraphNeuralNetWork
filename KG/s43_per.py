from tqdm import tqdm
import numpy as np
import collections
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, accuracy_score
import dataloader4kge, s42_pathSim
from sklearn.decomposition import NMF


# NFM矩阵分解
def getNFM(m, dim):
    nmf = NMF(n_components=dim)
    user_vectors = nmf.fit_transform(m)
    item_vectors = nmf.components_
    return user_vectors, item_vectors.T


# 将事实按照关系分开，代表不同的元路径
def splitTriples(kgTriples, movie_set):
    '''
    :param kgTriples: 知识图谱三元组
    :param movie_set: 包含所有电影的集合
    '''
    metapath_triples = {}
    for h, r, t in tqdm(kgTriples):
        if h in movie_set:
            h, r, t = int(h), int(r), int(t)
            if r not in metapath_triples:
                metapath_triples[r] = []
            metapath_triples[r].append([h, 1, t])
    return metapath_triples


# 得到所有元路径下的实体邻接表
def getAdjacencyListOfAllRelations(metapath_triples):
    print('得到所有元路径下的实体邻接表...')
    r_al = {}
    for r in tqdm(metapath_triples):
        r_al[r] = s42_pathSim.getAdjacencyListByTriples(metapath_triples[r])
    return r_al


# 得到所有元路径下的电影实体相似度矩阵
def getSimMatrixOfAllRelations(metapath_al, movie_set):
    print('计算实体相似度矩阵...')
    metapath_simMatrixs = {}
    for r in tqdm(metapath_al):
        metapath_simMatrixs[r] = \
            s42_pathSim.getSimMatrixFromAl(metapath_al[r], max(movie_set) + 1)
    return metapath_simMatrixs


class PER(nn.Module):

    def __init__(self, kgTriples, user_set, movie_set, recTriples, dim=8):
        super(PER, self).__init__()
        # 以不同关系作为不同的元路径切分知识图谱三元组事实
        metapath_triples = splitTriples(kgTriples, movie_set)
        # 根据切分好的三元组数据得到各个元路径下的邻接表
        metapath_al = getAdjacencyListOfAllRelations(metapath_triples)
        # 根据邻接表得到各个元路径下的路径相似度矩阵
        metapath_simMatrixs = getSimMatrixOfAllRelations(metapath_al, movie_set)
        print("计算用户偏好扩散矩阵...")
        sortedUserItemSims, self.metapath_map = self.init_userItemSims(user_set, recTriples, metapath_simMatrixs)
        print('初始化用户物品在每个元路径下的embedding...')
        self.embeddings = self.init_embedding(dim, sortedUserItemSims, self.metapath_map)

        # 用一个线性层来加载每个metapath所带的权重
        self.metapath_linear = nn.Linear(len(self.metapath_map), 1)

    # 初始化用户偏好扩散矩阵
    def init_userItemSims(self, user_set, recTriples, metapath_simMatrixs):
        # 根据推荐三元组数据得到用户物品邻接表
        userItemAl = s42_pathSim.getAdjacencyList(recTriples, r_col=2)

        userItemSims = collections.defaultdict(dict)
        for metapath in metapath_simMatrixs:
            for u in userItemAl:
                userItemSims[metapath][u] = \
                    np.sum(metapath_simMatrixs[metapath][[i for i in userItemAl[u] if userItemAl[u][i] == 1]], axis=0)

        userItemSimMatrixs = {}
        for metapath in tqdm(userItemSims):
            userItemSimMatrix = []
            for u in user_set:
                userItemSimMatrix.append(userItemSims[metapath][int(u)].tolist())
            userItemSimMatrixs[metapath] = np.mat(userItemSimMatrix)

        metapath_map = {k: v for k, v in enumerate(sorted([metapath for metapath in userItemSims]))}
        return userItemSimMatrixs, metapath_map

    # 初始化用户物品在每个元路径下的embedding
    def init_embedding(self, dim, sortedUserItemSims, metapath_map):
        embeddings = collections.defaultdict(dict)
        for metapath in metapath_map:
            # 根据NFM矩阵分解的方式得到用户特征表示及物品特征表示
            user_vectors, item_vectors = \
                self.__init_one_pre_emd(sortedUserItemSims[metapath_map[metapath]], dim)
            # 分别用先验的用户与物品的向量初始化每个元路径下代表表示用户及表示物品的embedding层
            embeddings[metapath]['user'] = \
                nn.Embedding.from_pretrained(user_vectors, max_norm=1)
            embeddings[metapath]['item'] = \
                nn.Embedding.from_pretrained(item_vectors, max_norm=1)
        return embeddings

    # 根据NFM矩阵分解的方式得到用户特征表示及物品特征表示
    def __init_one_pre_emd(self, mat, dim):
        user_vectors, item_vectors = getNFM(mat, dim)
        return torch.FloatTensor(user_vectors), torch.FloatTensor(item_vectors)

    def forward(self, u, v):
        metapath_preds = []
        for metapath in self.metapath_map:
            # [ batch_size, dim ]
            metapath_embs = self.embeddings[metapath]
            # [ batch_size, 1 ]
            metapath_pred = \
                torch.sum(metapath_embs['user'](u) *
                          metapath_embs['item'](v),
                          dim=1, keepdim=True)
            metapath_preds.append(metapath_pred)
        # [ batch_size, metapath_number ]
        metapath_preds = torch.cat(metapath_preds, 1)
        # [ batch_size, 1 ]
        metapath_preds = self.metapath_linear(metapath_preds)
        # [ batch_size ]
        logit = torch.sigmoid(metapath_preds).squeeze()
        return logit


def doEva(net, d):
    d = torch.LongTensor(d)
    u, i, r = d[:, 0], d[:, 1], d[:, 2]
    with torch.no_grad():
        out = net(u, i)
    y_pred = np.array([1 if i >= 0.6 else 0 for i in out])
    y_true = r.detach().numpy()
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    return p, r, acc


def train(epochs=10, batchSize=1024, lr=0.1, eva_per_epochs=1):
    entitys, relation, triples = dataloader4kge.readKGData()
    users, items, train_set, test_set = dataloader4kge.readRecData()

    net = PER(triples, users, items, train_set)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=1e-3)
    criterion = nn.BCELoss()

    for e in range(epochs):
        net.train()
        all_lose = 0
        for u, v, r in tqdm(DataLoader(train_set, batch_size=batchSize, shuffle=True)):
            r = torch.FloatTensor(r.detach().numpy())
            optimizer.zero_grad()
            result = net(u, v)
            loss = criterion(result, r)
            all_lose += loss
            loss.backward()
            optimizer.step()
        print('epoch {},avg_loss={:.4f}'.format(e, all_lose / len(train_set)))
        # 评估模型
        if e % eva_per_epochs == 0:
            p, r, acc = doEva(net, train_set)
            print('train: Precision {:.4f} | Recall {:.4f} | accuracy {:.4f}'.format(p, r, acc))
            p, r, acc = doEva(net, test_set)
            print('test: Precision {:.4f} | Recall {:.4f} | accuracy {:.4f}'.format(p, r, acc))


if __name__ == "__main__":
    train()
