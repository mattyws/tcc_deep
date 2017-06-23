'''
Hierarchy testing of the model
'''

import csv
import os
import pickle

import pandas as pd
import nltk
from random import shuffle

from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.pipeline import Pipeline
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

'''
See if the model exists
'''
def model_exists(modelName):
    # if not modelName+'.model' in os.listdir("./doc2vec_saved/"):
    #     return False
    if not modelName+'.model' in os.listdir("./NN_saved/"):
        return False
    return True

'''
Get doc2vec saved model with the name passed as a parameter
'''
def get_doc2vec_saved(modelName):
    # model = doc2vec.Doc2Vec.load("./doc2vec_saved/" + modelName + ".model")
    model = doc2vec.Doc2Vec.load("doc2vec_model.model")
    return model

'''
Get neural network saved model with the name passed as a parameter
'''
def get_NN_saved(modelName):
    model = joblib.load('./NN_saved/' + modelName +'.model')
    return model
'''
Get all data categories
'''
def get_categories(path):
    all_categories = os.listdir(path)
    # for dirName, subdirList, fileList in os.walk(path):
    # 	all_categories.append(dirName.split('/')[-1])
    all_categories.sort()
    return all_categories

'''
Return a list of all document side with their path
'''
def get_files_paths(path):
    all_files = []
    number = 0
    for dirName, subdirList, fileList in os.walk(path):
        for file in fileList:
            all_files.append(dirName+'/'+file)
    return all_files

class GetFilesFromPath(object):
    def __init__(self, path_list):
        self.path_list = path_list
    def __iter__(self):
        for path in self.path_list:
            cat = path.split("/")[-2]
            try:
                # Reading utf-8 file
                stream = open(path, encoding="UTF-8").read().replace("\n", " ").lower()
            except ValueError:
                # if error Read as ISO-8859-15 file
                stream = open(path, encoding="ISO-8859-15").read().replace("\n", " ").lower()
            yield [cat, stream, path.split('/')[-1]]

class LoadCorpus(object):
    def __init__(self, path_list, tokenizer, stop_set, stemmer):
        self.path_list = path_list
        self.tokenizer = tokenizer
        self.stop_set = stop_set
        self.stemmer = stemmer

    def __iter__(self):
        corpus = GetFilesFromPath(self.path_list)
        for data in corpus:
            text = TidenePreProcess.CleanStopWords(self.stop_set, TidenePreProcess.TokenizeFromList(self.tokenizer, [data[1]]))
            # text = TidenePreProcess.TokenizeFromList(self.tokenizer, [data[1]])
            for t in text:
                # print(t)
                t = [self.stemmer.stem(word) for word in t]
                # print(t)
                # exit(0)
                yield  doc2vec.TaggedDocument(t, [data[0]])
'''
Configurations
'''
firstModel = 'base'
language = 'english'
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
stop_set = nltk.corpus.stopwords.words(language)
stemmer = gensim.parsing.PorterStemmer()
pathname = 'testFiles2'
all_corpus = LoadCorpus(get_files_paths(pathname), tokenizer=tokenizer, stop_set=stop_set, stemmer=stemmer)

'''
Loading the first models
'''
doc2vecModel = get_doc2vec_saved(firstModel)
nnModel = get_NN_saved(firstModel)
# nnModel = Pipeline([("vectorizer", joblib.load('tfidf.model')), ("classifier", nnModel)])
'''
Loop through all files in the corpus to make the classification
'''
predictions = []
trueTags = []
print("Predicting tags from test files")
for data in all_corpus:
    i = 0
    trueTags.append(data[1])
    while i < 3:
        dataVector = doc2vecModel.infer_vector(data.words)
        pred = nnModel.predict(dataVector.reshape(1, -1))
        # pred = nnModel.predict([' '.join(data[0])])
        #print(data.tags, pred)
        i+=1
        if not i >= 3 and not model_exists(pred[0]) :
            print('Model do not exists ' + pred[0])
            predictions.append('')
            break
        if i != 3:
            # doc2vecModel = get_doc2vec_saved(pred[0])
            nnModel = get_NN_saved(pred[0])
        else:
            predictions.append(pred[0])
            print(data.tags, pred)
    # doc2vecModel = get_doc2vec_saved(firstModel)
    nnModel = get_NN_saved(firstModel)
    # nnModel = Pipeline([("vectorizer", joblib.load('tfidf.model')), ("classifier", nnModel)])

'''
Getting measures of the predictions
'''
document_recall = recall_score(trueTags, predictions, average='weighted')
document_precision = precision_score(trueTags, predictions, average='weighted')
document_f1 = f1_score(trueTags, predictions, average='weighted')
document_accuracy = accuracy_score(trueTags, predictions)
with open('hierarchcal.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    # writer.writerow(categories)
    # for row in document_matrix:
    # 	writer.writerow(row)
    writer.writerow(['Precision', document_precision])
    writer.writerow(['Recall', document_recall])
    writer.writerow(['Accuracy', document_accuracy])
    writer.writerow(['F1', document_f1])