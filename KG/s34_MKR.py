import dataloader4kge
from torch import nn
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score
from torch.nn import Parameter, init


class CrossCompress(nn.Module):

    def __init__(self, dim):
        super(CrossCompress, self).__init__()
        self.dim = dim

        self.weight_vv = init.xavier_uniform_(Parameter(torch.empty(dim, 1)))
        self.weight_ev = init.xavier_uniform_(Parameter(torch.empty(dim, 1)))
        self.weight_ve = init.xavier_uniform_(Parameter(torch.empty(dim, 1)))
        self.weight_ee = init.xavier_uniform_(Parameter(torch.empty(dim, 1)))

        self.bias_v = init.xavier_uniform_(Parameter(torch.empty(1, dim)))
        self.bias_e = init.xavier_uniform_(Parameter(torch.empty(1, dim)))

    def forward(self, v, e):
        # [ batch_size, dim ]
        # [ batch_size, dim, 1 ]
        v = v.reshape(-1, self.dim, 1)
        # [ batch_size, 1, dim ]
        e = e.reshape(-1, 1, self.dim)
        # [ batch_size, dim, dim ]
        c_matrix = torch.matmul(v, e)
        # [ batch_size, dim, dim ]
        c_matrix_transpose = torch.transpose(c_matrix, dim0=1, dim1=2)
        # [ batch_size * dim, dim ]
        c_matrix = c_matrix.reshape((-1, self.dim))
        c_matrix_transpose = c_matrix_transpose.reshape((-1, self.dim))
        # [batch_size, dim]
        v_output = torch.matmul(c_matrix, self.weight_vv) + torch.matmul(c_matrix_transpose, self.weight_ev)
        e_output = torch.matmul(c_matrix, self.weight_ve) + torch.matmul(c_matrix_transpose, self.weight_ee)
        # [batch_size, dim]
        v_output = v_output.reshape(-1, self.dim) + self.bias_v
        e_output = e_output.reshape(-1, self.dim) + self.bias_e
        return v_output, e_output


# 附加Dropout的全连接网络层
class DenseLayer(nn.Module):

    def __init__(self, in_dim, out_dim, dropout_prob):
        super(DenseLayer, self).__init__()
        self.liner = nn.Linear(in_dim, out_dim)
        self.drop = nn.Dropout(dropout_prob)

    def forward(self, x, isTrain):
        out = torch.relu(self.liner(x))
        if isTrain:  # 训练时加入dropout防止过拟合
            out = self.drop(out)
        return out


class MKR(nn.Module):

    def __init__(self, n_users, n_entitys, n_relations, dim=128, margin=1, alpha=0.2, dropout_prob=0.5):
        super().__init__()
        self.margin = margin
        self.u_emb = nn.Embedding(n_users, dim)  # 用户向量
        self.e_emb = nn.Embedding(n_entitys, dim)  # 实体向量
        self.r_emb = nn.Embedding(n_relations, dim)  # 关系向量

        self.user_dense1 = DenseLayer(dim, dim, dropout_prob)
        self.user_dense2 = DenseLayer(dim, dim, dropout_prob)
        self.user_dense3 = DenseLayer(dim, dim, dropout_prob)
        self.tail_dense1 = DenseLayer(dim, dim, dropout_prob)
        self.tail_dense2 = DenseLayer(dim, dim, dropout_prob)
        self.tail_dense3 = DenseLayer(dim, dim, dropout_prob)
        self.cc_unit1 = CrossCompress(dim)
        self.cc_unit2 = CrossCompress(dim)
        self.cc_unit3 = CrossCompress(dim)

        self.BCEloss = nn.BCELoss()

        self.alpha = alpha  # kge损失函数的计算权重

    def hinge_loss(self, y_pos, y_neg):
        dis = y_pos - y_neg + self.margin
        return torch.sum(torch.relu(dis))

    # kge采用最基础的TransE算法
    def TransE(self, h, r, t):
        score = h + r - t
        return torch.sum(score ** 2, dim=1) ** 0.5

    # 前向传播
    def forward(self, rec_set, kg_set, isTrain=True):
        # 推荐预测部分的提取初始embedding
        u, v, y = rec_set
        y = torch.FloatTensor(y.detach().numpy())
        u = self.u_emb(u)
        v = self.e_emb(v)

        # 分开知识图谱三元组的正负例
        x_pos, x_neg = kg_set

        # 提取知识图谱三元组正例h,r,t的初始embedding
        h_pos, r_pos, t_pos = x_pos
        h_pos = self.e_emb(h_pos)
        r_pos = self.r_emb(r_pos)
        t_pos = self.e_emb(t_pos)

        # 提取知识图谱三元组负例h,r,t的初始embedding
        h_neg, r_neg, t_neg = x_neg
        h_neg = self.e_emb(h_neg)
        r_neg = self.r_emb(r_neg)
        t_neg = self.e_emb(t_neg)

        # 将用户向量经三层全连接层传递
        u = self.user_dense1(u, isTrain)
        u = self.user_dense2(u, isTrain)
        u = self.user_dense3(u, isTrain)
        # 将KG正例的尾实体向量经三层全连接层传递
        t_pos = self.tail_dense1(t_pos, isTrain)
        t_pos = self.tail_dense2(t_pos, isTrain)
        t_pos = self.tail_dense3(t_pos, isTrain)

        # 将物品与KG正例头实体一同经三层C单元传递
        v, h_pos = self.cc_unit1(v, h_pos)
        v, h_pos = self.cc_unit2(v, h_pos)
        v, h_pos = self.cc_unit3(v, h_pos)

        # 计算推荐预测的预测值及损失函数
        rec_pred = torch.sigmoid(torch.sum(u * v, dim=1))
        rec_loss = self.BCEloss(rec_pred, y)

        # 计算kg正例的TransE评分
        kg_pos = self.TransE(h_pos, r_pos, t_pos)
        # 计算kg负例的TransE评分，注意负例的实体不要与物品向量一同去走C单元
        kg_neg = self.TransE(h_neg, r_neg, t_neg)
        # 计算kge的hing loss
        kge_loss = self.hinge_loss(kg_pos, kg_neg)

        # 将推荐产生的损失函数与kge产生的损失函数加权相加后返回
        return rec_loss + self.alpha * kge_loss

    # 测试时用
    def predict(self, u, v, isTrain=False):
        u = self.u_emb(u)
        v = self.e_emb(v)
        u = self.user_dense1(u, isTrain)
        u = self.user_dense2(u, isTrain)
        u = self.user_dense3(u, isTrain)
        # 第一层输入C单元的KG头实体是物品自身
        v, h = self.cc_unit1(v, v)
        v, h = self.cc_unit2(v, h)
        v, h = self.cc_unit3(v, h)
        return torch.sigmoid(torch.sum(u * v, dim=1))


# 预测
def doEva(net, d):
    d = torch.LongTensor(d)
    u, i, r = d[:, 0], d[:, 1], d[:, 2]
    with torch.no_grad():
        out = net.predict(u, i, False)
    y_pred = np.array([1 if i >= 0.5 else 0 for i in out])
    y_true = r.detach().numpy()
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    return p, r, acc


def train(epochs=20, batchSize=1024, lr=0.01, dim=128, eva_per_epochs=1):
    # 读取数据
    entitys, relation, triples = dataloader4kge.readKGData()
    kgTrainSet = dataloader4kge.KgDatasetWithNegativeSampling(triples, entitys)
    users, items, train_set, test_set = dataloader4kge.readRecData()

    # 初始化模型
    net = MKR(max(users) + 1, max(entitys) + 1, max(relation) + 1, dim)
    # 初始化优化器
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=5e-3)
    # 开始训练
    for e in range(epochs):
        net.train()
        all_loss = 0
        # 同时采样用户物品三元组及知识图谱三元组数据, 但因为C单元中物品与头实体的计算过程相互干涉，所以batch_size必须一致
        for rec_set, kg_set in tqdm(zip(DataLoader(train_set, batch_size=batchSize, shuffle=True, drop_last=True),
                                        DataLoader(kgTrainSet, batch_size=batchSize, shuffle=True, drop_last=True))):
            optimizer.zero_grad()
            loss = net(rec_set, kg_set)
            all_loss += loss
            loss.backward()
            optimizer.step()
        print('epoch {},avg_loss={:.4f}'.format(e, all_loss / (len(train_set))))

        # 评估模型
        if e % eva_per_epochs == 0:
            p, r, acc = doEva(net, train_set)
            print('train: Precision {:.4f} | Recall {:.4f} | accuracy {:.4f}'.format(p, r, acc))
            p, r, acc = doEva(net, test_set)
            print('test: Precision {:.4f} | Recall {:.4f} | accuracy {:.4f}'.format(p, r, acc))


if __name__ == '__main__':
    train()
