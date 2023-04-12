import torch
from torch import nn
from chapter5 import dataloader4kge
from torch.utils.data import DataLoader
from tqdm import tqdm

class TransH( nn.Module ):

    def __init__( self, n_entitys, n_relations, dim = 128, margin = 1 ):
        super( ).__init__( )
        self.margin = margin # hinge_loss中的差距
        self.n_entitys = n_entitys #实体的数量
        self.n_relations = n_relations #关系的数量
        self.dim = dim #embedding的长度

        # 随机初始化实体的embedding
        self.e = nn.Embedding( self.n_entitys, dim, max_norm = 1 )
        # 随机初始化关系的embedding
        self.r = nn.Embedding( self.n_relations, dim, max_norm = 1 )
        # 随机初始化法向量的embedding
        self.wr = nn.Embedding(self.n_relations, dim, max_norm = 1)


    def forward( self, X ):
        x_pos, x_neg = X
        y_pos = self.predict( x_pos )
        y_neg = self.predict( x_neg )
        return self.hinge_loss( y_pos, y_neg )

    def predict( self, x ):
        h, r_index, t = x
        h = self.e( h )
        r = self.r( r_index )
        t = self.e( t )
        wr = self.wr( r_index )
        score = self.Htransfer( h, wr ) + r - self.Htransfer( t, wr )
        return torch.sum( score**2, dim = 1 )**0.5

    def Htransfer( self, e, wr ):
        return e - torch.sum( e * wr, dim = 1, keepdim = True ) * wr

    def hinge_loss( self, y_pos, y_neg ):
        dis = y_pos - y_neg + self.margin
        return torch.sum( torch.relu( dis ) )

def train( epochs = 20, batchSize = 1024, lr = 0.01, dim = 128 ):
    #读取数据
    entitys, relation, triples = dataloader4kge.readKGData( )
    train_set = dataloader4kge.KgDatasetWithNegativeSampling( triples, entitys )
    #初始化模型
    net = TransH( max( entitys ) + 1 , max( relation ) + 1, dim )
    #初始化优化器
    optimizer = torch.optim.AdamW( net.parameters(), lr = lr, weight_decay = 5e-3 )
    #开始训练
    for e in range(epochs):
        net.train()
        all_lose = 0
        for X in tqdm( DataLoader( train_set, batch_size = batchSize, shuffle = True )):
            optimizer.zero_grad( )
            loss = net( X )
            all_lose += loss
            loss.backward( )
            optimizer.step( )
        print('epoch {},avg_loss={:.4f}'.format( e, all_lose/( len( triples ))))


if __name__ == '__main__':
    train()


