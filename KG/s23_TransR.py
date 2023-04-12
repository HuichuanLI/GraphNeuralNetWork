import torch
from torch import nn
import dataloader4kge
from torch.utils.data import DataLoader
from tqdm import tqdm


class TransR(nn.Module):

    def __init__(self, n_entitys, n_relations, k_dim=128, r_dim=64, margin=1):
        super().__init__()
        self.margin = margin  # hinge_loss中的差距
        self.n_entitys = n_entitys  # 实体的数量
        self.n_relations = n_relations  # 关系的数量
        self.k_dim = k_dim  # 实体embedding的长度
        self.r_dim = r_dim  # 关系embedding的长度

        # 随机初始化实体的embedding
        self.e = nn.Embedding(self.n_entitys, k_dim)
        # 随机初始化关系的embedding
        self.r = nn.Embedding(self.n_relations, r_dim)
        # 随机初始化变换矩阵
        self.Mr = nn.Embedding(self.n_relations, k_dim * r_dim)

    def forward(self, X):
        x_pos, x_neg = X
        y_pos = self.predict(x_pos)
        y_neg = self.predict(x_neg)
        return self.hinge_loss(y_pos, y_neg)

    def predict(self, x):
        h, r_index, t = x
        h = self.e(h)
        r = self.r(r_index)
        t = self.e(t)
        mr = self.Mr(r_index)
        score = self.Rtransfer(h, mr) + r - self.Rtransfer(t, mr)
        return torch.sum(score ** 2, dim=1) ** 0.5

    def Rtransfer(self, e, mr):
        # [ batch_size, 1, e_dim ]
        e = torch.unsqueeze(e, dim=1)
        # [ batch_size, e_dim, r_dim ]
        mr = mr.reshape(-1, self.k_dim, self.r_dim)
        # [ batch_size, 1, r_dim ]
        result = torch.matmul(e, mr)
        # [ batch_size, r_dim ]
        result = torch.squeeze(result)
        return result

    def hinge_loss(self, y_pos, y_neg):
        dis = y_pos - y_neg + self.margin
        return torch.sum(torch.relu(dis))


def train(epochs=20, batchSize=1024, lr=0.01, dim=128):
    # 读取数据
    entitys, relation, triples = dataloader4kge.readKGData()
    train_set = dataloader4kge.KgDatasetWithNegativeSampling(triples, entitys)
    # 初始化模型
    net = TransR(max(entitys) + 1, max(relation) + 1, dim)
    # 初始化优化器
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=5e-3)
    # 开始训练
    for e in range(epochs):
        net.train()
        all_lose = 0
        for X in tqdm(DataLoader(train_set, batch_size=batchSize, shuffle=True)):
            optimizer.zero_grad()
            loss = net(X)
            all_lose += loss
            loss.backward()
            optimizer.step()
        print('epoch {},avg_loss={:.4f}'.format(e, all_lose / (len(triples))))


if __name__ == '__main__':
    train()
