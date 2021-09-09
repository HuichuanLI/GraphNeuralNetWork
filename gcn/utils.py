# -*- coding:utf-8 -*-
# @Time : 2021/9/5 4:06 下午
# @Author : huichuan LI
# @File : utils.py
# @Software: PyCharm

import numpy as np
import scipy.sparse as sp


def encode_onehot(labels):
    classes = set(labels)
    # identity创建方矩阵
    # 字典key为label的值，value为矩阵的每一行
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    # get函数得到字典key对应的value
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

    # map() 会根据提供的函数对指定序列做映射
    # 第一个参数 function 以参数序列中的每一个元素调用 function 函数，返回包含每次 function 函数返回值的新列表
    #  map(lambda x: x ** 2, [1, 2, 3, 4, 5])
    #  output:[1, 4, 9, 16, 25]


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    # https://towardsdatascience.com/how-to-do-deep-learning-on-graphs-with-graph-convolutional-networks-7d2250723780
    # https://towardsdatascience.com/understanding-graph-convolutional-networks-for-node-classification-a2bfdb7aba7b
    # 论文里A^=(D~)^0.5 A~ (D~)^0.5这个公式
    # 对每一行求和
    rowsum = np.array(mx.sum(1))
    # (D~)^0.5
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    # 如果某一行全为0，则r_inv算出来会等于无穷大，将这些行的r_inv置为0
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    # 构建对角元素为r_inv的对角矩阵
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    # 论文里A^=(D~)^0.5 A~ (D~)^0.5这个公式
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def get_splits(y, ):
    idx_list = np.arange(len(y))
    # train_val, idx_test = train_test_split(idx_list, test_size=0.2, random_state=1024)  # 1000
    # idx_train, idx_val = train_test_split(train_val, test_size=0.2, random_state=1024)  # 500

    idx_train = []
    label_count = {}
    for i, label in enumerate(y):
        label = np.argmax(label)
        if label_count.get(label, 0) < 20:
            idx_train.append(i)
            label_count[label] = label_count.get(label, 0) + 1

    idx_val_test = list(set(idx_list) - set(idx_train))
    idx_val = idx_val_test[0:500]
    idx_test = idx_val_test[500:1500]

    y_train = np.zeros(y.shape, dtype=np.int32)
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_test = np.zeros(y.shape, dtype=np.int32)
    y_train[idx_train] = y[idx_train]
    y_val[idx_val] = y[idx_val]
    y_test[idx_test] = y[idx_test]
    train_mask = sample_mask(idx_train, y.shape[0])
    val_mask = sample_mask(idx_val, y.shape[0])
    test_mask = sample_mask(idx_test, y.shape[0])

    return y_train, y_val, y_test, train_mask, val_mask, test_mask


def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def convert_symmetric(X, sparse=True):
    if sparse:
        X += X.T - sp.diags(X.diagonal())
    else:
        X += X.T - np.diag(X.diagonal())
    return X


def load_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # 由于文件中节点并非是按顺序排列的，因此建立一个编号为0-(node_size-1)的哈希表idx_map，
    # 哈希表中每一项为old id: number，即节点id对应的编号为number
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)

    idx_map = {j: i for i, j in enumerate(idx)}

    # edges_unordered为直接从边表文件中直接读取的结果，是一个(edge_num, 2)的数组，每一行表示一条边两个端点的idx
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)

    # flatten：降维，返回一维数组
    # 边的edges_unordered中存储的是端点id，要将每一项的old id换成编号number
    # 在idx_map中以idx作为键查找得到对应节点的编号，reshape成与edges_unordered形状一样的数组
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)

    # 根据coo矩阵性质，这一段的作用就是，网络有多少条边，邻接矩阵就有多少个1，
    # 所以先创建一个长度为edge_num的全1数组，每个1的填充位置就是一条边中两个端点的编号，
    # 即edges[:, 0], edges[:, 1]，矩阵的形状为(node_size, node_size)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    # 对于无向图，邻接矩阵是对称的。上一步得到的adj是按有向图构建的，转换成无向图的邻接矩阵需要扩充成对称矩阵
    # 将i->j与j->i中权重最大的那个, 作为无向图的节点i与节点j的边权.
    # https://blog.csdn.net/Eric_1993/article/details/102907104
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = convert_symmetric(adj, )

    features = normalize_features(features)

    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    # idx_train = range(140)
    # idx_val = range(200, 500)
    # idx_test = range(500, 1500)
    y_train, y_val, y_test, train_mask, val_mask, test_mask = get_splits(labels)

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
