import pandas as pd
from torch.utils.data import DataLoader
import dataloader4graph
from tqdm import tqdm
import torch
from torch import nn
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score


class GAFM(torch.nn.Module):

    def __init__(self, n_users, n_entitys, k_dim, t_dim, atten_way):

        super(GAFM, self).__init__()

        self.entitys = nn.Embedding(n_entitys, k_dim, max_norm=1)
        self.users = nn.Embedding(n_users, k_dim, max_norm=1)

        self.a_liner = nn.Linear(k_dim, t_dim)
        self.h_liner = nn.Linear(t_dim, 1)

        # base, item, user
        self.atten_way = atten_way

    # FM聚合
    def FMaggregator(self, feature_embs):
        # feature_embs:[ batch_size, n_features, k ]
        # [ batch_size, k ]
        square_of_sum = torch.sum(feature_embs, dim=1) ** 2
        # [ batch_size, k ]
        sum_of_square = torch.sum(feature_embs ** 2, dim=1)
        # [ batch_size, k ]
        output = square_of_sum - sum_of_square
        return output

    # 根据上一轮聚合的输出向量，原始索引与记录原始索引与更新后索引的映射表得到这一阶的输入邻居节点向量
    def __getEmbeddingByNeibourIndex(self, orginal_indexes, nbIndexs, aggEmbeddings):
        new_embs = []
        for v in orginal_indexes:
            embs = aggEmbeddings[torch.squeeze(torch.LongTensor(nbIndexs.loc[v].values))]
            new_embs.append(torch.unsqueeze(embs, dim=0))
        return torch.cat(new_embs, dim=0)

    # 注意力计算
    def attention(self, embs, target_embs=None):
        # embs: [ batch_size, k ]
        # target_embs : [batch_size, k]
        if target_embs != None:
            embs = target_embs * embs
        # [ batch_size, t ]
        embs = self.a_liner(embs)
        # [ batch_size, t ]
        embs = torch.relu(embs)
        # [ batch_size, 1 ]
        embs = self.h_liner(embs)
        # [ batch_size, 1 ]
        atts = torch.softmax(embs, dim=1)
        return atts

    def gnnForward(self, adj_lists, user_embs=None):
        n_hop = 0
        for df in adj_lists:
            if n_hop == 0:
                entity_embs = self.entitys(torch.LongTensor(df.values))
            else:
                entity_embs = self.__getEmbeddingByNeibourIndex(df.values, neighborIndexs, aggEmbeddings)
            target_embs = self.entitys(torch.LongTensor(df.index))
            aggEmbeddings = self.FMaggregator(entity_embs)
            if self.atten_way == 'item':
                # item参与注意力计算 [batch_size, dim]
                atts = self.attention(aggEmbeddings, target_embs)
                if n_hop < len(adj_lists):
                    neighborIndexs = pd.DataFrame(range(len(df.index)), index=df.index)
            elif self.atten_way == 'user':
                if n_hop < len(adj_lists):
                    neighborIndexs = pd.DataFrame(range(len(df.index)), index=df.index)
                    # 最后一层之前的注意力仍然采用item形式的即可
                    atts = self.attention(aggEmbeddings, target_embs)
                else:
                    # 用户的向量参与注意力计算[ batch_size, dim ]
                    atts = self.attention(aggEmbeddings, user_embs)
            else:
                atts = self.attention(aggEmbeddings)
                if n_hop < len(adj_lists):
                    neighborIndexs = pd.DataFrame(range(len(df.index)), index=df.index)

            aggEmbeddings = atts * aggEmbeddings + target_embs
            n_hop += 1
        # [ batch_size, dim ]
        return aggEmbeddings

    def forward(self, u, adj_lists):
        # [batch_size, dim]
        items = self.gnnForward(adj_lists)
        # [batch_size, dim]
        users = self.users(u)
        # [batch_size]
        uv = torch.sum(users * items, dim=1)
        # [batch_size]
        logit = torch.sigmoid(uv)
        return logit


@torch.no_grad()
def doEva(net, d, G):
    net.eval()
    d = torch.LongTensor(d)
    u, i, r = d[:, 0], d[:, 1], d[:, 2]
    i_index = i.detach().numpy()
    adj_lists = dataloader4graph.graphSage4RecAdjType(G, i_index)
    out = net(u, adj_lists)
    y_pred = np.array([1 if i >= 0.5 else 0 for i in out])
    y_true = r.detach().numpy()
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    return p, r, acc


def train(epoch=20, batchSize=1024, dim=128, tdim=64, lr=0.002, eva_per_epochs=1, atten_way='base'):
    user_set, item_set, train_set, test_set = dataloader4graph.readRecData()
    entitys, pairs = dataloader4graph.readGraphData()
    G = dataloader4graph.get_graph(pairs)
    net = GAFM(max(user_set) + 1, max(entitys) + 1, dim, tdim, atten_way)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)

    for e in range(epoch):
        net.train()
        all_lose = 0
        for u, i, r in tqdm(DataLoader(train_set, batch_size=batchSize, shuffle=True)):
            r = torch.FloatTensor(r.detach().numpy())
            optimizer.zero_grad()
            i_index = i.detach().numpy()
            adj_lists = dataloader4graph.graphSage4RecAdjType(G, i_index)
            logits = net(u, adj_lists)
            loss = criterion(logits, r)
            all_lose += loss
            loss.backward()
            optimizer.step()

        print('epoch {}, avg_loss = {:.4f}'.format(e, all_lose / (len(train_set) // batchSize)))

        # 评估模型
        if e % eva_per_epochs == 0:
            p, r, acc = doEva(net, train_set, G)
            print('train: Precision {:.4f} | Recall {:.4f} | accuracy {:.4f}'.format(p, r, acc))
            p, r, acc = doEva(net, test_set, G)
            print('test: Precision {:.4f} | Recall {:.4f} | accuracy {:.4f}'.format(p, r, acc))
    return net


if __name__ == '__main__':
    train(atten_way='user')
