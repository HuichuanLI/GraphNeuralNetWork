import networkx as nx
import numpy as np
from tqdm import tqdm
import dataloader4kge
from gensim.models import word2vec


# 将事实按照关系分开，代表不同的元路径
def splitTriples(kgTriples):
    '''
    :param kgTriples: 知识图谱三元组
    '''
    metapath_pairs = {}
    for h, r, t in tqdm(kgTriples):
        if r not in metapath_pairs:
            metapath_pairs[r] = []
        metapath_pairs[r].append([h, t])
    return metapath_pairs


# 根据边集生成networkx的有向图图
def get_graph(pairs):
    G = nx.Graph()
    G.add_edges_from(pairs)  # 通过边集加载数据
    return G


def fromTriplesGeneralSubGraphSepByMetaPath(triples):
    '''
    :param triples: 知识图谱三元组信息
    :return: 各个元路径的networkx子图
    '''
    metapath_pairs = splitTriples(triples)
    graphs = []
    for metapath in metapath_pairs:
        graphs.append(get_graph(metapath_pairs[metapath]))
    return graphs


# 随机游走生成序列
def getDeepwalkSeqs(g, walk_length, num_walks):
    seqs = []
    for _ in tqdm(range(num_walks)):
        start_node = np.random.choice(g.nodes)
        w = walkOneTime(g, start_node, walk_length)
        seqs.append(w)
    return seqs


# 一次随机游走
def walkOneTime(g, start_node, walk_length):
    walk = [str(start_node)]  # 初始化游走序列
    for _ in range(walk_length):  # 最大长度范围内进行采样
        current_node = int(walk[-1])
        neighbors = list(g.neighbors(current_node))  # 获取当前节点的邻居
        if len(neighbors) > 0:
            next_node = np.random.choice(neighbors, 1)
            walk.extend([str(n) for n in next_node])
    return walk


def multi_metaPath2vec(graphs, dim=16, walk_length=12, num_walks=256, min_count=3):
    seqs = []
    for g in graphs:
        # 将不同元路径随机游走生成的序列合并起来
        seqs.extend(getDeepwalkSeqs(g, walk_length, num_walks))
    model = word2vec.Word2Vec(seqs, size=dim, min_count=min_count)
    return model


if __name__ == '__main__':
    # 读取知识图谱数据
    _, _, triples = dataloader4kge.readKGData()
    graphs = fromTriplesGeneralSubGraphSepByMetaPath(triples)
    model = multi_metaPath2vec(graphs)
    print(model.wv.most_similar('259', topn=3))  # 观察与节点259最相近的三个节点
    model.wv.save_word2vec_format('e.emd')  # 可以把emd储存下来以便下游任务使用
    model.save('m.model')  # 可以把模型储存下来以便下游任务使用
