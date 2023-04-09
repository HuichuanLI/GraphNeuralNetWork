import pandas as pd
from torch.utils.data import DataLoader
import dataloader4graph
from tqdm import tqdm
import torch
from torch import nn
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score


class GAT4Rec(torch.nn.Module):

    def __init__(self, n_users, n_entitys, dim):

        super(GAT4Rec, self).__init__()

        self.entitys = nn.Embedding(n_entitys, dim, max_norm=1)
        self.users = nn.Embedding(n_users, dim, max_norm=1)

        self.multiHeadNumber = 2

        self.W = nn.Linear(in_features=dim, out_features=dim // self.multiHeadNumber, bias=False)
        self.a = nn.Linear(in_features=dim, out_features=1, bias=False)
        self.leakyRelu = nn.LeakyReLU(negative_slope=0.2)

    def oneHeadAttention(self, target_embeddings, neighbor_entitys_embeddings):
        # [ batch_size, w_dim ]
        target_embeddings_w = self.W(target_embeddings)
        # [ batch_size, n_neighbor, w_dim ]
        neighbor_entitys_embeddings_w = self.W(neighbor_entitys_embeddings)
        # [ batch_size, n_neighbor, w_dim ]
        target_embeddings_broadcast = torch.cat(
            [torch.unsqueeze(target_embeddings_w, 1)
             for _ in range(neighbor_entitys_embeddings.shape[1])], dim=1)
        # [ batch_size, n_neighbor, w_dim*2 ]
        cat_embeddings = torch.cat([target_embeddings_broadcast, neighbor_entitys_embeddings_w], dim=-1)
        # [ batch_size, n_neighbor, 1 ]
        eijs = self.leakyRelu(self.a(cat_embeddings))
        # [ batch_size, n_neighbor, 1 ]
        aijs = torch.softmax(eijs, dim=1)
        # [ batch_size, w_dim]
        out = torch.sum(aijs * neighbor_entitys_embeddings_w, dim=1)
        return out

    def multiHeadAttentionAggregator(self, target_embeddings, neighbor_entitys_embeddings):
        '''
        :param target_embeddings: 目标节点的向量 [ batch_size, dim ]
        :param neighbor_entitys_embeddings: 目标节点的邻居节点向量 [ batch_size, n_neighbor, dim ]
        '''
        embs = []
        for i in range(self.multiHeadNumber):  # 循环多头注意力的头数
            embs.append(self.oneHeadAttention(target_embeddings, neighbor_entitys_embeddings))
        # 将每次单头注意力层得到的输出张量拼接后输出
        return torch.cat(embs, dim=-1)

    # 根据上一轮聚合的输出向量，原始索引与记录原始索引与更新后索引的映射表得到这一阶的输入邻居节点向量
    def __getEmbeddingByNeibourIndex(self, orginal_indexes, nbIndexs, aggEmbeddings):
        new_embs = []
        for v in orginal_indexes:
            embs = aggEmbeddings[torch.squeeze(torch.LongTensor(nbIndexs.loc[v].values))]
            new_embs.append(torch.unsqueeze(embs, dim=0))
        return torch.cat(new_embs, dim=0)

    def gnnForward(self, adj_lists):
        n_hop = 0
        for df in adj_lists:
            if n_hop == 0:
                # 最外阶的聚合可直接通过初始索引提取 [ 图采样时的某一阶的中心节点数量, neibours, dim ]
                entity_embs = self.entitys(torch.LongTensor(df.values))
            else:
                '''第二次开始聚合的邻居向量是第一次聚合后得到的，所以不能直接用self.entitys去提取，
                而是应该用上一次的聚合输出aggEmbeddings来提取节点向量表示。但图采样记录的节点索引
                对应的是self.entitys的节点索引，无法通过该索引直接提取出aggEmbeddings中对应的向量，
                所以需要一个记录初始索引映射到更新后索引的映射表neighbourIndexs。通过这些内容提取向
                量的具体操作可详见self.__getEmbeddingByNeibourIndex()这个方法'''
                entity_embs = self.__getEmbeddingByNeibourIndex(df.values, neighborIndexs, aggEmbeddings)
            # [图采样时的某一阶的中心节点数量, dim ]
            target_embs = self.entitys(torch.LongTensor(df.index))
            if n_hop < len(adj_lists):
                neighborIndexs = pd.DataFrame(range(len(df.index)), index=df.index)
            # 将得到的目标节点向量与其邻居节点向量传入GAT的多头注意力层聚合出更新后的目标节点向量
            # [图采样时的某一阶的中心节点数量, dim ]
            aggEmbeddings = self.multiHeadAttentionAggregator(target_embs, entity_embs)
            n_hop += 1
        # 返回最后的目标节点向量也就是指定代表这一批次的物品向量,形状为 [ batch_size, dim ]
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


def train(epoch=20, batchSize=1024, dim=128, lr=0.01, eva_per_epochs=1):
    user_set, item_set, train_set, test_set = dataloader4graph.readRecData()
    entitys, pairs = dataloader4graph.readGraphData()
    G = dataloader4graph.get_graph(pairs)
    net = GAT4Rec(max(user_set) + 1, max(entitys) + 1, dim)

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
            print(logits)
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
    train()
