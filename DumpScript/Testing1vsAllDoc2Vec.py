'''
https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/test/test_doc2vec.py
https://rare-technologies.com/word2vec-tutorial/
'''

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

'''
See if the model exists
'''
def model_exists(modelName):
    if not modelName+'.model' in os.listdir("./doc2vec_saved/"):
        return False
    if not modelName+'.model' in os.listdir("./NN_saved/"):
        return False
    return True

'''
Get doc2vec saved model with the name passed as a parameter
'''
def get_doc2vec_saved(modelName):
    model = doc2vec.Doc2Vec.load("./doc2vec_saved/" + modelName + ".model")
    return model

'''
Get neural network saved model with the name passed as a parameter
'''
def get_NN_saved(modelName):
    model = joblib.load('./NN_saved/' + modelName +'.model')
    return model
def dbow(alpha, min_alpha):
    """Test DBOW doc2vec training."""
    model = doc2vec.Doc2Vec(alpha=alpha, min_alpha=min_alpha, dm=0, hs=1, negative=0,
                            min_count=2, window=2)
    return 'dbow', model
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

'''
Class that loads the patents based on their classification. Initiate the class using the Subclass classification path.
The data must be in a hierarchcal structure of classes.
'''
class HierarchicalStructure(object):
    def __init__(self, path_list):
        self.path_list = path_list

    '''
    Get a dict of files, where each key is a section in the patent IPC classification (A-H)
    '''
    def __sections(self):
        section_dict = dict()
        for path in  self.path_list:
            print(path.split('/')[-2][0])
            if not path.split('/')[-2][0] in section_dict.keys():
                section_dict[path.split('/')[-2][0]] = []
            section_dict[path.split('/')[-2][0]].append(path)
        return section_dict

    '''
    Get a dic of files, where each key is a class in the patent IPC classification (The patent's section + two numbers)
    '''
    def __classes(self):
        class_dict = dict()
        for path in self.path_list:
            if not path.split('/')[-3] in class_dict.keys():
                class_dict[path.split('/')[-3]] = []
            class_dict[path.split('/')[-3]].append(path)
        return class_dict

    '''
    Get a dict of files, where each key is a subclass in the patent IPC classification (The patent's section + the patent's
    class + a letter (A-Z)
    '''
    def __subclasses(self):
        subclass_dict = dict()
        for path in self.path_list:
            if not path.split('/')[-2] in subclass_dict.keys():
                subclass_dict[path.split('/')[-2]] = []
            subclass_dict[path.split('/')[-2]].append(path)
        return subclass_dict

    def __iter__(self):
        for level in  [self.__sections(), self.__classes(), self.__subclasses()]:
            yield level

class GetFilesFromPath(object):
    def __init__(self, key, path_dict):
        self.path_dict = path_dict
        self.key = key
    def __iter__(self):
        # for key in self.path_dict.keys():
        #     for path in self.path_dict[key]:
        for path in self.path_dict[self.key]:
            try:
                # Reading utf-8 file
                stream = open(path, encoding="UTF-8").read().replace("\n", " ").lower()
            except ValueError:
                # if error Read as ISO-8859-15 file
                stream = open(path, encoding="ISO-8859-15").read().replace("\n", " ").lower()
            yield [self.key, stream]
                # if key == self.key:
                #     yield ['True', stream]
                # else:
                #     yield ['False', stream]

class LoadCorpus(object):
    def __init__(self, key, path_dict, tokenizer, stop_set, stemmer):
        self.path_dict = path_dict
        self.tokenizer = tokenizer
        self.stop_set = stop_set
        self.stemmer = stemmer
        self.key = key

    def __iter__(self):
        corpus = GetFilesFromPath(self.key, self.path_dict)
        for data in corpus:
            text = TidenePreProcess.CleanStopWords(self.stop_set, TidenePreProcess.TokenizeFromList(self.tokenizer, [data[1]]))
            # text = TidenePreProcess.TokenizeFromList(self.tokenizer, [data[1]])
            for t in text:
                # print(t)
                t = [self.stemmer.stem(word) for word in t]
                # print(t)
                # exit(0)
                yield doc2vec.TaggedDocument(t, [data[0]])
'''
Configurations
'''
firstModel = 'base'
language = 'english'
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
stop_set = nltk.corpus.stopwords.words(language)
stemmer = gensim.parsing.PorterStemmer()
pathname = 'testFiles2'
# all_corpus = LoadCorpus(get_files_paths(pathname), tokenizer=tokenizer, stop_set=stop_set, stemmer=stemmer)

structure = HierarchicalStructure(get_files_paths(pathname))

predictions = []
trueTags = []
for level in structure:
    for key in level:
        all_corpus = LoadCorpus(key, level, tokenizer=tokenizer, stop_set=stop_set, stemmer=stemmer)
        if model_exists(key):
            doc2vecModel = get_doc2vec_saved(key)
            nnModel = get_NN_saved(key)
            for data in all_corpus:
                trueTags.append(data.tags[0])
                inferedVector = doc2vecModel.infer_vector(data.words)
                pred = nnModel.predict(inferedVector.reshape(1, -1))
                print(key, pred)
                if pred == 'True':
                    predictions.append(key)
                else:
                    predictions.append('')
        else:
            break
    break

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