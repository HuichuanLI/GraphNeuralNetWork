import torch
import torch.nn.functional as F
import pygDataLoader
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):

    def __init__(self, n_classes, dim):
        '''
        :param n_classes: 类别数
        :param dim: 特征维度
        '''
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dim, 16)
        self.conv2 = GCNConv(16, n_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


@torch.no_grad()
def eva(net, data):
    net.eval()
    logits, accs = net(data), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


def train(epochs=200, lr=0.01):
    data, n_class, dim = pygDataLoader.loadData()
    net = GCN(n_class, dim)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)

    for epoch in range(epochs):
        net.train()
        optimizer.zero_grad()
        logits = net(data)
        # 仅用训练集计算loss
        loss = F.nll_loss(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        train_acc, val_acc, test_acc = eva(net, data)

        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, train_acc, val_acc, test_acc))


if __name__ == '__main__':
    train()
