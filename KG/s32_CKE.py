import dataloader4kge
from torch import nn
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score


class CKE(nn.Module):

    def __init__(self, n_users, n_entitys, n_relations, e_dim=128, margin=1, alpha=0.2):
        super().__init__()
        self.margin = margin
        self.u_emb = nn.Embedding(n_users, e_dim)  # 用户向量
        self.e_emb = nn.Embedding(n_entitys, e_dim)  # 实体向量
        self.r_emb = nn.Embedding(n_relations, e_dim)  # 关系向量

        self.BCEloss = nn.BCELoss()

        self.alpha = alpha  # kge损失函数的计算权重

    def hinge_loss(self, y_pos, y_neg):
        dis = y_pos - y_neg + self.margin
        return torch.sum(torch.relu(dis))

    # kge采用最基础的TransE算法
    def kg_predict(self, x):
        h, r, t = x
        h = self.e_emb(h)
        r = self.r_emb(r)
        t = self.e_emb(t)
        score = h + r - t
        return torch.sum(score ** 2, dim=1) ** 0.5

    # 计算kge损失函数
    def calculatingKgeLoss(self, kg_set):
        x_pos, x_neg = kg_set
        y_pos = self.kg_predict(x_pos)
        y_neg = self.kg_predict(x_neg)
        return self.hinge_loss(y_pos, y_neg)

    # 推荐采取最简单的ALS算法
    def rec_predict(self, u, i):
        u = self.u_emb(u)
        i = self.e_emb(i)
        y = torch.sigmoid(torch.sum(u * i, dim=1))
        return y

    # 计算推荐损失函数
    def calculatingRecLoss(self, rec_set):
        u, i, y = rec_set
        y_pred = self.rec_predict(u, i)
        y = torch.FloatTensor(y.detach().numpy())
        return self.BCEloss(y_pred, y)

    # 前向传播
    def forward(self, rec_set, kg_set):
        rec_loss = self.calculatingRecLoss(rec_set)
        kg_loss = self.calculatingKgeLoss(kg_set)
        # 分别得到推荐产生的损失函数与kge产生的损失函数加权相加后返回
        return rec_loss + self.alpha * kg_loss


# 预测
def doEva(net, d):
    d = torch.LongTensor(d)
    u, i, r = d[:, 0], d[:, 1], d[:, 2]
    with torch.no_grad():
        out = net.rec_predict(u, i)
    y_pred = np.array([1 if i >= 0.5 else 0 for i in out])
    y_true = r.detach().numpy()
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    return p, r, acc


def train(epochs=20, rec_batchSize=1024, kg_batchSize=512, lr=0.01, dim=128, eva_per_epochs=1):
    # 读取数据
    entitys, relation, triples = dataloader4kge.readKGData()
    kgTrainSet = dataloader4kge.KgDatasetWithNegativeSampling(triples, entitys)
    users, items, train_set, test_set = dataloader4kge.readRecData()

    # 初始化模型
    net = CKE(max(users) + 1, max(entitys) + 1, max(relation) + 1, dim)
    # 初始化优化器
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=5e-3)
    # 开始训练
    for e in range(epochs):
        net.train()
        all_lose = 0
        # 同时采样用户物品三元组及知识图谱三元组数据, 因计算过程中互相独立所以batch_size可设成不一样的值
        for rec_set, kg_set in tqdm(zip(DataLoader(train_set, batch_size=rec_batchSize, shuffle=True),
                                        DataLoader(kgTrainSet, batch_size=kg_batchSize, shuffle=True))):
            optimizer.zero_grad()
            loss = net(rec_set, kg_set)
            all_lose += loss
            loss.backward()
            optimizer.step()
        print('epoch {},avg_loss={:.4f}'.format(e, all_lose / (len(train_set))))

        # 评估模型
        if e % eva_per_epochs == 0:
            p, r, acc = doEva(net, train_set)
            print('train: Precision {:.4f} | Recall {:.4f} | accuracy {:.4f}'.format(p, r, acc))
            p, r, acc = doEva(net, test_set)
            print('test: Precision {:.4f} | Recall {:.4f} | accuracy {:.4f}'.format(p, r, acc))


if __name__ == '__main__':
    train()
