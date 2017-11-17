import csv
import os
import pickle

import pandas as pd
import nltk
from random import shuffle

from sklearn.externals import joblib
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.utils import shuffle
import numpy as np
import TideneLoadCorpus
import TideneTextNormalize
import TidenePreProcess
import TideneMisc

import gensim
from gensim.models import doc2vec
from sklearn.cross_validation import *
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.metrics.classification import f1_score, classification_report, accuracy_score

import pymongo

from DeepLearning.database import MongoLoadDocumentMeta, MongoLoadDocumentData

client = pymongo.MongoClient()
patents_database = client.patents

training_collection = patents_database.doc2vec_docs
docs = training_collection.find()

subclass_ipc = set()
for doc in docs:
    if len(doc['ipc_classes']) > 0 :
        subclass_ipc.add(doc['ipc_classes'][0])
    else:
        print(doc['filename'])

section_ipc = set()
class_ipc = set()
for subclass in subclass_ipc:
    section_ipc.add(subclass[0])
    class_ipc.add(subclass[0:3])


print(section_ipc)
print(class_ipc)

'''
Configurations
'''
ALPHA = 0.025
MIN_ALPHA=0.001
splitRateTrainPerc = 80
splitRateTestPerc = 20
randomInt = 42
language = 'english'
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
stop_set = nltk.corpus.stopwords.words(language)
stemmer = gensim.parsing.PorterStemmer()
mongodb = MongoLoadDocumentMeta('patents')
documents = mongodb.get_all_meta('doc2vec_docs')
corpus = MongoLoadDocumentData('patents', documents, clean_text=True, tokenizer=tokenizer, stop_set=stop_set,
                                   description=True)

for document in documents:
    print(document['filename'])
    model = doc2vec.Doc2Vec.load("doc2vec_mongo.model")
    # content = corpus.get_file_content(document['filename'])
    # content = corpus.clean(content['description'])
    # client = pymongo.MongoClient()
    # patents_database = client.patents
    # word_embedding_collection = patents_database.documents_embedding_docs100
    # document['embedding'] = bson.binary.Binary(pickle.dumps(word_embedding_matrix, protocol=2))
    # word_embedding_collection.insert_one(document)
    # fs = gridfs.GridFS(patents_database, collection="documents_embedding_docs100")
    # with fs.new_file(filename=document['filename'], content_type="binary") as fp:
    #     fp.write(word_embedding_matrix)
    # print(content)


# print("Preparing data for neural network for " + pathname)
# model = doc2vec.Doc2Vec.load("doc2vec_saved/"+pathname.split('/')[-1]+".model")
# files_paths = get_files_paths(pathname)
# data_table = []
# classes_list = []
# for path in files_paths:
#     data = LoadCorpus([path], tokenizer=tokenizer, stop_set=stop_set, stemmer=stemmer)
#     for d in data:
#         data_table.append([model.infer_vector(d.words), path.split('/')[-2]])
# data_table = shuffle(data_table)
# x = np.array([x[0] for x in data_table])
# y = np.array([y[1] for y in data_table])
#
# kf = KFold(10, shuffle=True, random_state=None)
#
# print(" 10-fold Cross-Validation training and testing \n")
#
# i = 1
#
# tableResults = []
# tableResults=[]
# NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10), random_state=1)
# for trainIndex, testIndex in kf.split(x):
#     print(" ============== Fold ", i, "============\n")
#     trainDocs, testDocs = x[trainIndex], x[testIndex]
#     trainCats, testCats = y[trainIndex], y[testIndex]
#     NN.fit(trainDocs, trainCats)
#     pred = NN.predict(testDocs)
#     accuracy = accuracy_score(testCats, pred)
#     recall = recall_score(testCats, pred, average='weighted')
#     precision = precision_score(testCats, pred, average='weighted')
#     f1 = f1_score(testCats, pred, average='weighted')
#     tableResults.append({'model': 'NN', 'accuracy': accuracy, 'recall': recall, 'precision': precision, 'f1': f1})
#     i+=1
# joblib.dump(NN, 'NN_saved/' + pathname.split('/')[-1]+'.model')
# NN = joblib.load('NN_saved/' + pathname.split('/')[-1]+'.model')
# measures = ['precision', 'recall', 'accuracy', 'f1']
# with open('neuralNN' + pathname.replace('/', '_')+'.csv', 'w') as f:
#     writer = csv.writer(f, delimiter=',')
#     df = pd.DataFrame(tableResults)
#     filt = pd.pivot_table(df, values=['precision', 'recall', 'accuracy', 'f1'], index=['model'])
#     print(" Results")
#     print(filt)
#     for measure in measures:
#         value = [measure]
#         value.append(filt[measure]['NN'])
#         writer.writerow(value)