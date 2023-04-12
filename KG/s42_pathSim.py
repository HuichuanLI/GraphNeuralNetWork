import numpy as np
import collections

triples=[[0,5,0],
       [0,2,1],
       [1,2,0],
       [1,3,1],
       [1,5,2],
       [2,10,2],
       [3,5,0],
       [3,2,1]]

M=np.mat([[5,2,0],
         [2,3,5],
         [0,0,10],
         [5,2,0]])

#根据共现矩阵求两个实体间的路径相似度
def getPathSimFromCoMatrix(e1,e2,M):
    CM = np.array(M.dot(M.T))#得到交换矩阵
    return 2*CM[e1][e2]/(CM[e1][e1]+CM[e2][e2])

#根据点乘的方法求两个实体间的路径相似度
def getPathSimFromMatrix(e1,e2,M):
    up=2*M[e1].dot(M[e2].T)
    down=M[e1].dot(M[e1].T)+M[e2].dot(M[e2].T)
    return float(up/down)
#根据共现矩阵得到所有实体的相似度矩阵
def getSimMatrixFromCoMatrix(M):
    CM=M.dot(M.T)
    a=np.diagonal(CM)
    nm=np.array([a+i for i in a])
    return 2*CM/nm

#根据三元组得到邻接表
def getAdjacencyListByTriples( triples ):
    al = collections.defaultdict( dict )
    for h, r, t in triples:
        al[h][t]=r
    return al

#根据三元组得到邻接表( 可指定关系列 )
def getAdjacencyList( triples, r_col=1 ):
    al = collections.defaultdict( dict )
    for p in triples:
        h = int( p[0] )
        r = int( p[r_col] )
        t = int( p[2]) if r_col == 1 else int( p[1] )
        if t not in al[h]:al[h][t] = 0
        al[h][t] += r
    return al

#得到自元路径数量
def getSelfMetaPathCount(e,al):
    return sum(al[e][i]**2 for i in al[e])

#得到两个实体间的元路径数
def getMetaPathCountBetween(e1,e2,al):
    return sum([al[e1][i]*al[e2][i] for i in set(al[e1]) & set(al[e2])])

#求两个实体间的路径相似度
def getPathSimFromAl(e1,e2,al):
    up=getMetaPathCountBetween(e1,e2,al)
    s1=getSelfMetaPathCount(e1,al)
    s2=getSelfMetaPathCount(e2,al)
    down=s1+s2
    return 2*up/down

#根据邻接表求所有实体间的路径相似度
def getSimMatrixFromAl(al,n_e):
    selfMPC = {}
    for e in al:
        selfMPC[e] = getSelfMetaPathCount( e, al )
    simMatrix=np.zeros( ( n_e, n_e ) )
    for e1 in al:
        for e2 in al:
            simMatrix[e1][e2]=2*getMetaPathCountBetween(e1,e2,al)\
                              /(selfMPC[e1]+selfMPC[e2])
    return simMatrix


if __name__=='__main__':
    print(getPathSimFromCoMatrix(1,0,M))
    print(getPathSimFromMatrix(0,1,M))

    print(getSimMatrixFromCoMatrix(M))
    print(getAdjacencyList(triples))
    print(getSimMatrixFromAl(getAdjacencyList(triples),4))


