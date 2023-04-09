from torch_geometric.datasets import Planetoid
import os


def loadData():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'Cora')
    dataset = Planetoid(path, 'Cora')
    return dataset.data, dataset.num_classes, dataset.num_features


if __name__ == '__main__':
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'Cora')
    dataset = Planetoid(path, 'Cora')
    print('类别数:', dataset.num_classes)
    print('特征维度:', dataset.num_features)
    print(dataset.data)
    print(dataset.data.train_mask)

    print(sum(dataset.data.train_mask))
    print(sum(dataset.data.val_mask))
    print(sum(dataset.data.test_mask))
