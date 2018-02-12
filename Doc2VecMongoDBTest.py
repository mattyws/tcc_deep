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
import DeepLearning as dl


'''
Configurations
'''
language = 'english'
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
stop_set = nltk.corpus.stopwords.words(language)
stemmer = gensim.parsing.PorterStemmer()
mongodb = MongoLoadDocumentMeta('patents')
models = [
    # '../doc2vec_models/doc2vec_mongo_50.model',
    '../doc2vec_models/doc2vec_mongo_200.model',
    '../doc2vec_models/doc2vec_mongo_300.model',
    '../doc2vec_models/doc2vec_mongo_400.model',
    '../doc2vec_models/doc2vec_old_50.model',
    '../doc2vec_models/doc2vec_old_200.model',
    '../doc2vec_models/doc2vec_old_300.model',
    '../doc2vec_models/doc2vec_old_400.model'
]

for model in models:
    real = []
    pred = []
    i = 0
    documents = mongodb.get_all_meta('testing_docs100')
    corpus = MongoLoadDocumentData('patents', documents, clean_text=True, tokenizer=tokenizer, stop_set=stop_set,description=True)
    doc2vec_model = dl.learn.Doc2VecTrainer().load_model(model)
    doc_vector_generator = dl.data_representation.Doc2VecEmbeddingCreator(doc2vec_model)
    for doc in documents:
        content = corpus.get_file_content(doc['filename'])
        content = corpus.clean(content['description'])
        if i%1000 == 0:
            print(str(i) + ' ' + doc['filename'])
        doc_embedding_vector = doc_vector_generator.create_x_text(content)#.reshape((1,embedding_size))
        pred.append(doc2vec_model.docvecs.most_similar([doc_embedding_vector])[0][0]) #adding the result to the predicted vector
        real.append(doc['ipc_classes'][0][0])
        i+=1
    accuracy = accuracy_score(real, pred)
    print("Model " + model + " accuracy: " +str(accuracy))
