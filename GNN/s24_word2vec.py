from gensim.models import word2vec

#
# s1=['今天','天气','真好']
# s2=['今天','天气','很好']

s1 = [1, 2, 3]
s2 = [1, 2, 4]

seqs = [s1, s2]

model = word2vec.Word2Vec(seqs, vector_size=10, min_count=1)
print(model.wv.most_similar(1, topn=3))
