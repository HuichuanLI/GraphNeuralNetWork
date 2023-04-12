import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm #产生进度条
from chapter5 import dataloader4kge, dataloader4KGNN
from utils import evaluate
from torch.utils.data import DataLoader

class KGCN( nn.Module ):

    def __init__( self, n_users, n_entitys, n_relations,
                  e_dim,  adj_entity, adj_relation, n_neighbors,
                  aggregator_method = 'sum',
                  act_method = F.relu, drop_rate = 0.5):
        super( KGCN, self ).__init__()

        self.e_dim = e_dim  # 特征向量维度
        self.aggregator_method = aggregator_method #消息聚合方法
        self.n_neighbors = n_neighbors #邻居的数量
        self.user_embedding = nn.Embedding( n_users, e_dim, max_norm = 1 )
        self.entity_embedding = nn.Embedding( n_entitys, e_dim, max_norm = 1)
        self.relation_embedding = nn.Embedding( n_relations, e_dim, max_norm = 1)

        self.adj_entity = adj_entity #节点的邻接列表
        self.adj_relation = adj_relation #关系的邻接列表

        #线性层
        self.linear_layer = nn.Linear(
                in_features = self.e_dim * 2 if self.aggregator_method == 'concat' else self.e_dim,
                out_features = self.e_dim, bias = True)

        self.act = act_method #激活函数
        self.drop_rate = drop_rate #drop out 的比率

    def forward( self, users, items, is_evaluate = False ):
        user_embeddings = self.user_embedding( users)
        item_embeddings = self.entity_embedding( items )
        # 得到邻居实体和连接它们关系的embedding
        neighbor_entitys, neighbor_relations = self.get_neighbors( items )
        # 得到v波浪线
        neighbor_vectors = self.__get_neighbor_vectors( neighbor_entitys, neighbor_relations, user_embeddings )
        # 聚合得到物品向量
        out_item_embeddings = self.aggregator( item_embeddings, neighbor_vectors, is_evaluate )

        out = torch.sigmoid( torch.sum( user_embeddings * out_item_embeddings, dim = -1 ) )

        return out

    # 得到邻居的节点embedding和关系embedding
    def get_neighbors( self, items ):
        e_ids = [self.adj_entity[ item ] for item in items ]
        r_ids = [ self.adj_relation[ item ] for item in items ]
        e_ids = torch.LongTensor( e_ids )
        r_ids = torch.LongTensor( r_ids )
        neighbor_entities_embs = self.entity_embedding( e_ids )
        neighbor_relations_embs = self.relation_embedding( r_ids )
        return neighbor_entities_embs, neighbor_relations_embs

    # 得到v波浪线
    def __get_neighbor_vectors(self, neighbor_entitys, neighbor_relations, user_embeddings):
        # [batch_size, n_neighbor, dim]
        user_embeddings = torch.cat([ torch.unsqueeze( user_embeddings, 1 ) for _ in range( self.n_neighbors ) ], dim = 1 )
        # [batch_size, n_neighbor]
        user_relation_scores = torch.sum( user_embeddings * neighbor_relations, dim = 2 )
        # [batch_size, n_neighbor]
        user_relation_scores_normalized = F.softmax( user_relation_scores, dim = -1 )
        # [batch_size, n_neighbor, 1]
        user_relation_scores_normalized = torch.unsqueeze( user_relation_scores_normalized, 2 )
        # [batch_size, dim ]
        neighbor_vectors = torch.sum( user_relation_scores_normalized * neighbor_entitys, dim = 1 )
        return neighbor_vectors

    #经过进一步的聚合与线性层得到v
    def aggregator(self,item_embeddings, neighbor_vectors, is_evaluate):
        # [batch_size, dim]
        if self.aggregator_method == 'sum':
            output = item_embeddings + neighbor_vectors
        elif self.aggregator_method == 'concat':
            # [batch_size, dim * 2]
            output = torch.cat( [ item_embeddings, neighbor_vectors ], dim = -1 )
        else:#neighbor
            output = neighbor_vectors
        if not is_evaluate:
            output = F.dropout( output, self.drop_rate )
        # [batch_size, dim]
        output = self.linear_layer( output )
        return self.act( output )

#验证
def doEva( model, testSet ):
    testSet = torch.LongTensor(testSet)
    model.eval()
    with torch.no_grad():
        user_ids = testSet[:, 0]
        item_ids = testSet[:, 1]
        labels = testSet[:, 2]
        logits = model( user_ids, item_ids, True )
        predictions = [ 1 if i >= 0.5 else 0 for i in logits ]
        p = evaluate.precision( y_true = labels, y_pred = predictions )
        r = evaluate.recall( y_true = labels, y_pred = predictions )
        acc = evaluate.accuracy_score( labels, y_pred = predictions )
        return p,r,acc

def train( epochs = 20, batchSize = 1024, lr = 0.01, dim = 128, n_neighbors=10, eva_per_epochs=1):


    users, items, train_set, test_set = dataloader4kge.readRecData()
    entitys, relations, kgTriples = dataloader4kge.readKGData()
    kg_indexes = dataloader4KGNN.getKgIndexsFromKgTriples(kgTriples)

    adj_entity, adj_relation = dataloader4KGNN.construct_adj(n_neighbors, kg_indexes, len(entitys))
    net = KGCN( max(users)+1, max(entitys)+1, max(relations)+1,
                  dim, adj_entity, adj_relation,n_neighbors = n_neighbors)
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
    train(  )



