import gensim
import nltk
from sklearn.metrics.classification import accuracy_score, recall_score, precision_score, f1_score

import DeepLearning as dl
import numpy as np

doc2vec_model = dl.learn.Doc2VecTrainer().load_model('doc2vec_model.model')
train_data = dl.database.FlatStructureDatabase('../database/descriptions/descriptions50')
level_subclass = train_data.subclasses()

tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
stop_set = nltk.corpus.stopwords.words('english')
data = dl.database.LoadTextCorpus(level_subclass, tokenizer=tokenizer, stop_set=stop_set)

real = []
pred = []
for i in range(0, len(data)):
    d = next(data)
    real.append(d.tags[0])
    vec = doc2vec_model.infer_vector(d.words)
    p = doc2vec_model.docvecs.most_similar([vec], topn=10)
    print(d.tags[0], p)
    if d.tags[0] in p:
        print(d.tags[0] + " in " + p)
        pred.append(d.tags[0])
    else:
        pred.append(p[0])