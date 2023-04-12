import torch
import torch.nn as nn
from tqdm import tqdm #产生进度条
from chapter5 import dataloader4kge, dataloader4KGNN
from utils import evaluate
from torch.utils.data import DataLoader

class KGFM( nn.Module ):

    def __init__( self, n_users, n_entitys, n_relations, dim,
                  adj_entity, adj_relation ,agg_method = 'Bi-Interaction',
                  mp_method = 'FM_KGCN'):
        super( KGFM, self ).__init__( )

        self.user_embs = nn.Embedding( n_users, dim, max_norm = 1 )
        self.entity_embs = nn.Embedding( n_entitys, dim, max_norm = 1 )
        self.relation_embs = nn.Embedding( n_relations, dim, max_norm = 1 )

        self.adj_entity = adj_entity  # 节点的邻接列表
        self.adj_relation = adj_relation  # 关系的邻接列表

        self.agg_method = agg_method # 聚合方法
        self.mp_method = mp_method # 消息传递方法

        # 初始化计算注意力时的关系变换线性层
        self.Wr = nn.Linear( dim, dim )

        # 初始化最终聚合时所用的激活函数
        self.leakyRelu = nn.LeakyReLU( negative_slope = 0.2 )

        # 初始化各种聚合时所用的线性层
        if agg_method == 'concat':
            self.W_concat = nn.Linear( dim * 2, dim )
        else:
            self.W1 = nn.Linear( dim, dim )
            if agg_method == 'Bi-Interaction':
                self.W2 = nn.Linear( dim, dim )

    # 得到邻居的节点embedding和关系embedding
    def get_neighbors( self, items ):
        e_ids = [self.adj_entity[ item ] for item in items ]
        r_ids = [ self.adj_relation[ item ] for item in items ]
        e_ids = torch.LongTensor( e_ids )
        r_ids = torch.LongTensor( r_ids )
        neighbor_entities_embs = self.entity_embs( e_ids )
        neighbor_relations_embs = self.relation_embs( r_ids )
        return neighbor_entities_embs, neighbor_relations_embs

    # KGAT由来的FM消息传递
    def FMMessagePassFromKGAT(self, h_embs, r_embs, t_embs ):
        '''
        :param h_embs: 头实体向量[ batch_size, dim ]
        :param r_embs: 关系向量[ batch_size, n_neibours, dim ]
        :param t_embs: 为实体向量[ batch_size, n_neibours, dim ]
        '''
        # # 将h张量广播，维度扩散为 [ batch_size, n_neibours, dim ]
        h_broadcast_embs = torch.cat( [ torch.unsqueeze( h_embs, 1 ) for _ in range( t_embs.shape[ 1 ] ) ], dim = 1 )
        # [ batch_size, n_neibours, dim ]
        tr_embs = self.Wr( t_embs )
        # [ batch_size, n_neibours, dim ]
        hr_embs = self.Wr( h_broadcast_embs )
        # [ batch_size, n_neibours, dim ]
        hr_embs= torch.tanh( hr_embs + r_embs)
        # [ batch_size, n_neibours, dim ]
        hrt_embs = hr_embs * tr_embs
        # [ batch_size, dim ]
        square_of_sum = torch.sum(hrt_embs, dim=1) ** 2
        # [ batch_size, dim ]
        sum_of_square = torch.sum(hrt_embs ** 2, dim=1)
        # [ batch_size, dim ]
        output = square_of_sum - sum_of_square
        return output

    # KGCN由来的FM消息聚合
    def FMMessagePassFromKGCN( self, u_embs, r_embs, t_embs ):
        '''
        :param u_embs: 用户向量[ batch_size, dim ]
        :param r_embs: 关系向量[ batch_size, n_neibours, dim ]
        :param t_embs: 为实体向量[ batch_size, n_neibours, dim ]
        '''
        # 将用户张量广播，维度扩散为 [ batch_size, n_neibours, dim ]
        u_broadcast_embs = torch.cat( [ torch.unsqueeze( u_embs, 1 ) for _ in range( t_embs.shape[ 1 ] ) ], dim = 1 )
        # [ batch_size, n_neighbor ]
        ur_embs = torch.sum( u_broadcast_embs * r_embs, dim = 2 )
        # [ batch_size, n_neighbor ]
        ur_embs = torch.softmax(ur_embs, dim=-1)
        # [ batch_size, n_neighbor, 1 ]
        ur_embs = torch.unsqueeze( ur_embs, 2 )
        # [ batch_size, n_neighbor, dim ]
        t_embs = ur_embs * t_embs
        # [ batch_size, dim ]
        square_of_sum = torch.sum( t_embs, dim = 1 ) ** 2
        # [ batch_size, dim ]
        sum_of_square = torch.sum( t_embs ** 2, dim = 1)
        # [ batch_size, dim ]
        output = square_of_sum - sum_of_square
        return output

    # GAT消息传递
    def GATMessagePass( self, h_embs, r_embs, t_embs ):
        '''
        :param h_embs: 头实体向量[ batch_size, n_neibours, dim ]
        :param r_embs: 关系向量[ batch_size, n_neibours, dim ]
        :param t_embs: 为实体向量[ batch_size, n_neibours, dim ]
        '''
        # [ batch_size, n_neibours, dim ]
        tr_embs = self.Wr( t_embs )
        # [ batch_size, n_neibours, dim ]
        hr_embs = self.Wr( h_embs )
        # [ batch_size, n_neibours, dim ]
        hr_embs= torch.tanh( hr_embs + r_embs)
        # [ batch_size, n_neibours, 1]
        atten = torch.sum( hr_embs * tr_embs,dim = -1 ,keepdim=True)
        atten = torch.softmax( atten, dim = -1 )
        # [ batch_size, n_neibours, dim]
        t_embs = t_embs * atten
        # [ batch_size, dim ]
        return  torch.sum( t_embs, dim = 1 )

    # 消息聚合
    def aggregate( self, h_embs, Nh_embs, agg_method = 'Bi-Interaction' ):
        '''
        :param h_embs: 原始的头实体向量 [ batch_size, dim ]
        :param Nh_embs: 消息传递后头实体位置的向量 [ batch_size, dim ]
        :param agg_method: 聚合方式，总共有三种,分别是'Bi-Interaction','concat','sum'
        '''
        if agg_method == 'Bi-Interaction':
            return self.leakyRelu( self.W1( h_embs + Nh_embs ) )\
                   + self.leakyRelu( self.W2( h_embs * Nh_embs ) )
        elif agg_method == 'concat':
            return self.leakyRelu( self.W_concat( torch.cat([ h_embs,Nh_embs ], dim = -1 ) ) )
        else: #sum
            return self.leakyRelu( self.W1( h_embs + Nh_embs ) )

    def forward( self, u, i ):
        # # [ batch_size, n_neibours, dim ] and # [ batch_size, n_neibours, dim ]
        t_embs, r_embs = self.get_neighbors( i )
        # # [ batch_size, dim ]
        h_embs = self.entity_embs( i )
        # # [ batch_size, dim ]
        user_embs = self.user_embs( u )
        # # [ batch_size, dim ]
        if self.mp_method =='FM_KGCN':
            Nh_embs = self.FMMessagePassFromKGCN( user_embs, r_embs, t_embs )
        else:
            Nh_embs = self.FMMessagePassFromKGAT( h_embs, r_embs, t_embs )
        # # [ batch_size, dim ]
        item_embs = self.aggregate( h_embs, Nh_embs, self.agg_method )
        # # [ batch_size ]
        logits = torch.sigmoid( torch.sum( user_embs * item_embs, dim = 1 ) )
        return logits


#验证
def doEva( model, testSet ):
    testSet = torch.LongTensor( testSet )
    model.eval( )
    with torch.no_grad( ):
        user_ids = testSet[:, 0]
        item_ids = testSet[:, 1]
        labels = testSet[:, 2]
        logits = model( user_ids, item_ids )
        predictions = [ 1 if i >= 0.5 else 0 for i in logits ]
        p = evaluate.precision( y_true = labels, y_pred = predictions )
        r = evaluate.recall( y_true = labels, y_pred = predictions )
        acc = evaluate.accuracy_score( labels, y_pred = predictions )
        return p,r,acc


def train( epochs = 20, batchSize = 1024, lr = 0.01, dim = 128, n_neighbors=10,eva_per_epochs=1 ):
    users, items, train_set, test_set = dataloader4kge.readRecData()
    entitys, relations, kgTriples = dataloader4kge.readKGData()
    kg_indexes = dataloader4KGNN.getKgIndexsFromKgTriples(kgTriples)

    adj_entity, adj_relation = dataloader4KGNN.construct_adj(n_neighbors, kg_indexes, len(entitys))

    net = KGFM( max( users ) + 1, max( entitys ) + 1, max( relations ) + 1 ,
                dim ,adj_entity, adj_relation)

    optimizer = torch.optim.Adam( net.parameters(), lr = lr, weight_decay = 5e-4 )
    loss_fcn = nn.BCELoss()

    print(len(train_set)//batchSize)

    for e in range( epochs ):
        net.train()
        all_loss = 0.0
        for u,i,r in tqdm( DataLoader( train_set, batch_size = batchSize,shuffle=True) ):
            logits = net( u, i )
            loss = loss_fcn( logits, r.float() )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            all_loss += loss.item()
        print('epoch {},avg_loss={:.4f}'.format(e, all_loss / (len(train_set) // batchSize)))

        # 评估模型
        if e % eva_per_epochs == 0:
            p, r, acc = doEva(net, train_set)
            print('train: Precision {:.4f} | Recall {:.4f} | accuracy {:.4f}'.format(p, r, acc))
            p, r, acc = doEva(net, test_set)
            print('test: Precision {:.4f} | Recall {:.4f} | accuracy {:.4f}'.format(p, r, acc))

if __name__ == '__main__':
    train()
