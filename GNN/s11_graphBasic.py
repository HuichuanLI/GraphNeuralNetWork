import networkx as nx
import matplotlib.pyplot as plt

edges = [(1, 2), (1, 5), (2, 4), (4, 3), (3, 1)]
G = nx.DiGraph()  # 初始化有向图
G.add_edges_from(edges)  # 通过边集加载数据

print(G.nodes)  # 打印所有节点
print(G.edges)  # 打印所有边

nx.draw(G)  # 画图
plt.show()  # 显示
