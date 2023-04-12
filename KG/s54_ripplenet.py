import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score,precision_score,recall_score
import numpy as np
from chapter5 import dataloader4KGNN, dataloader4kge
from torch.utils.data import DataLoader
from tqdm import tqdm

dim = 16 #实体和关系的向量维度
n_memory = 32 #每一波记录的节点数量
n_hop = 2 #水波扩撒的波数，该数字=水波层数-1
lr = 0.02 #学习率
batch_size = 1024 #批次数量
n_epoch = 20 #迭代次数
kge_weight = 0.01 #知识图谱嵌入损失函数系数


item_update_mode='plus_transform'
'''
物品向量的更新模式，总共有四种：
replace: 直接用新一波预测的物品向量替代
plus: 与t-1波次的物品向量对应位相加
replace_transform: 用一个映射矩阵映射将预测的物品向量映射一下
plus_transform: 用一个映射矩阵映射将预测的物品向量映射一下后与t-1波次的物品向量对应位相加
'''
using_all_hops = True # 最终用户向量的产生方式，是否采用全部波次的输出向量相加。否则采用最后一波产生的输出向量作为用户向量


class RippleNet(nn.Module):

    def __init__(self,n_entity,n_relation,dim=dim,n_hop=n_hop,
                 kge_weight=kge_weight,n_memory=n_memory,
                 item_update_mode=item_update_mode,using_all_hops=using_all_hops):
        super(RippleNet, self).__init__()

        self._parse_args(n_entity, n_relation,dim,n_hop,kge_weight,n_memory,item_update_mode,using_all_hops)
        self.entity_emb = nn.Embedding(self.n_entity, self.dim)
        self.relation_emb = nn.Embedding(self.n_relation, self.dim * self.dim)

        if item_update_mode == 'replace_transform' or item_update_mode == 'plus_transform':
            self.transform_matrix = nn.Linear(self.dim, self.dim, bias=False)

        self.criterion = nn.BCELoss()
        self.return_dict = {}

    def _parse_args(self, n_entity, n_relation,dim,n_hop,kge_weight,n_memory,item_update_mode,using_all_hops):
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.dim = dim
        self.n_hop = n_hop
        self.kge_weight = kge_weight
        self.n_memory = n_memory
        self.item_update_mode = item_update_mode
        self.using_all_hops = using_all_hops


    def _get_rec_loss( self, item_embs, labels, h_emb_list, r_emb_list, t_emb_list ):
        #o_list, item_embs = self._key_addressing(h_emb_list, r_emb_list, t_emb_list, item_embs)
        o_list = []
        for hop in range(self.n_hop):
            # [batch_size, n_memory, dim, 1]
            h_expanded = torch.unsqueeze(h_emb_list[hop], dim=3)
            # [batch_size, n_memory, dim]
            Rh = torch.squeeze(torch.matmul(r_emb_list[hop], h_expanded))
            # [batch_size, dim, 1]
            v = torch.unsqueeze(item_embs, dim=2)
            # [batch_size, n_memory]
            probs = torch.squeeze(torch.matmul(Rh, v))
            # [batch_size, n_memory]
            probs_normalized = F.softmax(probs, dim=1)
            # [batch_size, n_memory, 1]
            probs_expanded = torch.unsqueeze(probs_normalized, dim=2)
            # [batch_size, dim]
            o = (t_emb_list[hop] * probs_expanded).sum(dim=1)
            item_embs = self._update_item_embedding(item_embs, o)
            o_list.append(o)

        user_embs = self._get_user_embeddings(o_list)
        scores = torch.sigmoid((item_embs * user_embs).sum(dim=1))
        loss = self.criterion(scores, labels.float())
        self.return_dict['scores'] = scores
        return loss

    def forward(
        self,
        items: torch.LongTensor,
        labels: torch.LongTensor,
        memories_h: list,
        memories_r: list,
        memories_t: list,
    ):
        # [batch size, dim]
        item_embs = self.entity_emb(items)
        h_emb_list,r_emb_list,t_emb_list = [],[],[]

        for i in range(self.n_hop):
            # [batch size, n_memory, dim]
            h_emb_list.append(self.entity_emb(memories_h[i]))
            # [batch size, n_memory, dim, dim]
            r_emb_list.append(self.relation_emb(memories_r[i]).view(-1, self.n_memory, self.dim, self.dim))
            # [batch size, n_memory, dim]
            t_emb_list.append(self.entity_emb(memories_t[i]))

        rec_loss = self._get_rec_loss(item_embs, labels, h_emb_list, r_emb_list, t_emb_list)
        kge_loss = self._get_kg_loss(h_emb_list, r_emb_list, t_emb_list)
        self.return_dict['loss'] = rec_loss+kge_loss
        return self.return_dict

    # 产生kge loss来联合训练
    def _get_kg_loss( self, h_emb_list, r_emb_list, t_emb_list):
        '''
        h_emb_list,r_emb_list,t_emb_list是水波采样的实体与关系集，三者间位置是对应的
        '''
        kge_loss = 0
        for hop in range( self.n_hop ):
            # [batch size, n_memory, 1, dim]
            h_expanded = torch.unsqueeze( h_emb_list[hop], dim = 2 )
            # [batch size, n_memory, dim, 1]
            t_expanded = torch.unsqueeze( t_emb_list[hop], dim = 3 )
            # [batch size, n_memory, dim, dim]
            hRt = torch.squeeze(
                torch.matmul( torch.matmul( h_expanded, r_emb_list[ hop ] ), t_expanded )
            )
            kge_loss += torch.sigmoid( hRt ).mean( )
        kge_loss = -self.kge_weight * kge_loss
        return kge_loss

    # 迭代物品的向量
    def _update_item_embedding( self, item_embeddings, o ):
        '''
        :param item_embeddings: 上一个hop的物品向量 # [ batch_size, dim ]
        :param o: 当前hop的o向量 # [ batch_size, dim ]
        :return: 当前hop的物品向量 # [ batch_size, dim ]
        '''
        if self.item_update_mode == "replace":
            item_embeddings = o
        elif self.item_update_mode == "plus":
            item_embeddings = item_embeddings + o
        elif self.item_update_mode == "replace_transform":
            item_embeddings = self.transform_matrix( o )
        elif self.item_update_mode == "plus_transform":
            item_embeddings = self.transform_matrix( item_embeddings + o )
        else:
            raise Exception( "位置物品更新mode: " + self.item_update_mode )
        return item_embeddings

    def _get_user_embeddings( self, o_list ):
        '''
        :param o_list: 每一个hop得到的o向量集
        :return: 用户向量
        '''
        user_embs = o_list[-1]
        # 选择是否使用全部的o向量相加作为用户向量，否则仅用最后一层的o向量
        if self.using_all_hops:
            for i in range(self.n_hop - 1):
                user_embs += o_list[i]
        return user_embs

# 得到模型训练需要的数据
def get_feed_dict(data, ripple_set):
    u,i,r = data
    memories_h, memories_r, memories_t = [], [], []
    for hop in range(n_hop):
        memories_h.append(torch.LongTensor([ripple_set[int(user)][hop][0] for user in u]))
        memories_r.append(torch.LongTensor([ripple_set[int(user)][hop][1] for user in u]))
        memories_t.append(torch.LongTensor([ripple_set[int(user)][hop][2] for user in u]))
    return i, r, memories_h, memories_r,memories_t

# 生成单个用户的水波集
def get_single_user_ripple_set( single_user_history_dict, kg ):
    user_ripple=[]
    for h in range( n_hop ):
        memories_h = []
        memories_r = []
        memories_t = []
        if h == 0:
            tails_of_last_hop = single_user_history_dict
        else:
            tails_of_last_hop = user_ripple[-1][2]
        for entity in tails_of_last_hop:
            for tail_and_relation in kg.get(str(entity),[]):
                memories_h.append(entity)
                memories_r.append(tail_and_relation[1])
                memories_t.append(tail_and_relation[0])
        if len(memories_h) == 0:
            user_ripple.append(user_ripple[-1])
        else:
            replace = len(memories_h) < n_memory
            indices = np.random.choice(len(memories_h), size=n_memory, replace=replace)
            memories_h = [memories_h[i] for i in indices]
            memories_r = [memories_r[i] for i in indices]
            memories_t = [memories_t[i] for i in indices]
            user_ripple.append((memories_h, memories_r, memories_t))
    return user_ripple

# 生成水波集
def get_ripple_set(kg, user_history_dict):
    # user -> [(hop_0_heads, hop_0_relations, hop_0_tails), (hop_1_heads, hop_1_relations, hop_1_tails), ...]
    ripple_set = {}
    for user in user_history_dict:
        user_ripple = get_single_user_ripple_set(user_history_dict[user],kg)
        ripple_set[int(user)] = user_ripple
    return ripple_set

# 测试
def doEva(model, data_set, ripple_set, batch_size):
    auc_list,ps,rs = [],[],[]
    model.eval()
    for dataset in tqdm(DataLoader(data_set, batch_size = batch_size, shuffle=True)):
        items, labels, memories_h, memories_r, memories_t = get_feed_dict(dataset, ripple_set)
        return_dict = model(items, labels, memories_h, memories_r, memories_t)
        scores = return_dict["scores"].detach().cpu().numpy()
        labels = labels.cpu().numpy()
        predictions = [1 if i >= 0.5 else 0 for i in scores]

        p = precision_score(y_true=labels, y_pred=predictions)
        r = recall_score(y_true=labels, y_pred=predictions)
        auc = roc_auc_score(y_true=labels, y_score=scores)

        auc_list.append(auc)
        ps.append(p)
        rs.append(r)
    return float(np.mean(auc_list)), float(np.mean(ps)), float(np.mean(rs))


def train(n_epoch=n_epoch,batch_size=batch_size,eva_per_epochs=1):
    # 读取知识图谱数据
    entitys, relation, kg_triples = dataloader4kge.readKGData()
    # 根据知识图谱三元组数据得到知识图谱索引集
    kg_indexs = dataloader4KGNN.getKgIndexsFromKgTriples(kg_triples)
    # 读取用户物品三元组数据
    users, items, train_set, test_set = dataloader4kge.readRecData()
    # 读取用户正例集作为用户历史观看的物品
    user_history_pos_dict = dataloader4KGNN.getUserHistoryPosDict(train_set)
    # 将没有历史正例的用户过滤掉
    train_set = dataloader4KGNN.filetDateSet(train_set, user_history_pos_dict)
    test_set = dataloader4KGNN.filetDateSet(test_set, user_history_pos_dict)

    # 初始化模型与优化器
    net = RippleNet(max(entitys)+1, max(relation)+1)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),lr)

    # 开始训练
    for e in range(n_epoch):
        net.train()
        all_loss = 0
        # 每个epoch都重新生成水波集
        ripple_set = get_ripple_set(kg_indexs, user_history_pos_dict)
        for dataset in tqdm(DataLoader(train_set, batch_size=batch_size, shuffle=True)):
            return_dict = net(*get_feed_dict(dataset, ripple_set))
            loss = return_dict["loss"]
            all_loss+=loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('epoch {},avg_loss={:.4f}'.format(e, all_loss/(len(train_set)//batch_size)))

        # 评估模型
        if e % eva_per_epochs == 0:
            p, r, auc = doEva(net, train_set, ripple_set, batch_size)
            print('train: Precision {:.4f} | Recall {:.4f} | AUC {:.4f}'.format(p, r, auc))
            # 给测试集测试时重新生成水波集来增加预测难度
            ripple_set = get_ripple_set(kg_indexs, user_history_pos_dict)
            p, r, auc = doEva(net, test_set, ripple_set, batch_size)
            print('test: Precision {:.4f} | Recall {:.4f} | AUC {:.4f}'.format(p, r, auc))



if __name__=='__main__':
    train()