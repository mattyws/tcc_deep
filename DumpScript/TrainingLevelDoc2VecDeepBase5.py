'''
https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/test/test_doc2vec.py
https://rare-technologies.com/word2vec-tutorial/
'''
import csv
import os
import pickle

import keras
import pandas as pd
import nltk
from random import shuffle

from keras.engine.topology import Input
from keras.layers.convolutional import Convolution2D, Conv2D, Conv1D, Convolution1D
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.merge import Concatenate
from keras.layers.pooling import MaxPooling2D, MaxPooling1D
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from scipy.special._ufuncs import y0
from sklearn.externals import joblib
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.svm.classes import SVC
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
import tensorflow as tf


# '''
# Class that loads the patents based on their classification. Initiate the class using the Subclass classification path.
# The data must be in a hierarchcal structure of classes.
# '''
# class HierarchicalStructure(object):
#     def __init__(self, path_list):
#         self.path_list = path_list
#
#     '''
#     Get a dict of files, where each key is a section in the patent IPC classification (A-H)
#     '''
#     def __sections(self):
#         section_dict = dict()
#         for path in  self.path_list:
#             if not path.split('/')[-4] in section_dict.keys():
#                 section_dict[path.split('/')[-4]] = []
#             section_dict[path.split('/')[-4]].append(path)
#         return section_dict
#
#     '''
#     Get a dic of files, where each key is a class in the patent IPC classification (The patent's section + two numbers)
#     '''
#     def __classes(self):
#         class_dict = dict()
#         for path in self.path_list:
#             if not path.split('/')[-3] in class_dict.keys():
#                 class_dict[path.split('/')[-3]] = []
#             class_dict[path.split('/')[-3]].append(path)
#         return class_dict
#
#     '''
#     Get a dict of files, where each key is a subclass in the patent IPC classification (The patent's section + the patent's
#     class + a letter (A-Z)
#     '''
#     def __subclasses(self):
#         subclass_dict = dict()
#         for path in self.path_list:
#             if not path.split('/')[-2] in subclass_dict.keys():
#                 subclass_dict[path.split('/')[-2]] = []
#             subclass_dict[path.split('/')[-2]].append(path)
#         return subclass_dict
#
#     def __iter__(self):
#         for level in  [self.__sections(), self.__classes(), self.__subclasses()]:
#             yield level

class HierarchicalStructure(object):
    def __init__(self, path_list):
        self.path_list = path_list

    '''
    Get a dict of files, where each key is a section in the patent IPC classification (A-H)
    '''
    def __sections(self):
        section_dict = dict()
        for path in  self.path_list:
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
            if not path.split('/')[-2][0:3] in class_dict.keys():
                class_dict[path.split('/')[-2][0:3]] = []
            class_dict[path.split('/')[-2][0:3]].append(path)
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

def dbow(alpha, min_alpha):
    """Test DBOW doc2vec training."""
    model = doc2vec.Doc2Vec(alpha=alpha, min_alpha=min_alpha, dm=0, hs=1, negative=0,
                            min_count=2, window=2)
    return 'dbow', model

def separate_class(keys, level):
    data = []
    for key in keys:
        key_subclasses = dict()
        for lKey in level.keys():
            if lKey.startswith(key):
                key_subclasses[lKey] = level[lKey]
        data.append(key_subclasses)
    return data

def get_pathname(keys, dataP):
    if not keys:
        return 'base'
    for k in keys:
        for data_key in dataP.keys():
            if data_key.startswith(k):
                return k
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
    def __init__(self, key, path_dict):
        self.path_dict = path_dict
        self.key = key
    def __iter__(self):
        for key in self.path_dict.keys():
            for path in self.path_dict[key]:
                try:
                    # Reading utf-8 file
                    stream = open(path, encoding="UTF-8").read().replace("\n", " ").lower()
                except ValueError:
                    # if error Read as ISO-8859-15 file
                    stream = open(path, encoding="ISO-8859-15").read().replace("\n", " ").lower()
                yield [key, stream]

class LoadCorpus(object):
    def __init__(self, path_dict, tokenizer, stop_set, stemmer, key=None):
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

def get_models():
    models = dict()

    filter_sizes = (3, 8)
    dropout_prob = (0.5, 0.8)
    hidden_dims = 50

    models = dict()
    # Build model
    input_shape = (170, 1)
    model_input = Input(shape=input_shape)

    # Static model do not have embedding layer
    z = Dropout(dropout_prob[0])(model_input)

    # Convolutional block
    conv_blocks = []
    for sz in filter_sizes:
        conv = Convolution1D(filters=2,
                             kernel_size=sz,
                             padding="valid",
                             activation="relu",
                             strides=1)(z)
        conv = MaxPooling1D(pool_size=2)(conv)
        conv = Flatten()(conv)
        conv_blocks.append(conv)
    z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

    z = Dropout(dropout_prob[1])(z)
    z = Dense(hidden_dims, activation="relu")(z)
    model_output = Dense(7, activation="sigmoid")(z)

    models['convnet'] = keras.models.Model(model_input, model_output)
    models['convnet'].compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    return models

def create_directories(keys):
    for key in keys:
        if key+'_saved' not in os.listdir('./'):
            os.mkdir(key+'_saved')
        if key+'_results' not in os.listdir('./'):
            os.mkdir(key+'_results')
        # if 'result.csv' not in os.listdir('./'+key+'_results'):
        #     f = open('./'+key+'_results/result.csv', 'w')
        #     f.close()
def classMap(keys):
    i = 0
    map = dict()
    for k in keys:
        map[k] = i
        map[i] = k
        i+=1
    return map

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

'''
     Working path
'''
path = 'testFiles2'
# Creating the directories
create_directories(get_models().keys())

# Getting the hierarchcal structures from the database, and looping over it
structure = HierarchicalStructure(get_files_paths(path))
keys = None
for level in structure:
    class_map = classMap(level.keys())
    print(class_map)
    #If keys is not None, so we already train for the section level
    if keys:
        data_paths = separate_class(keys, level)
    else:
        data_paths = [level]
    #Looping through the files
    for dataP in data_paths:
        pathname = get_pathname(keys, dataP)
        print("Working on " + pathname)
        numDocs = 0
        for key in dataP.keys():
            numDocs += len(dataP[key])
        x = np.zeros(shape=(numDocs, 170, 1), dtype='int32')
        helper = np.zeros(shape=len(list(level.keys())), dtype='int32')
        y = np.zeros(shape=(numDocs))
        # if pathname not in os.listdir('trained'):
        print("Getting vectors for " + pathname)
        model = doc2vec.Doc2Vec.load("doc2vec_model.model")
        data_table = [[[]]]
        # y = []
        data = LoadCorpus(dataP, tokenizer=tokenizer, stop_set=stop_set, stemmer=stemmer)
        helper = 0
        for d in data:
            # data_table.append([model.infer_vector(d.words), d.tags[0]])
            for enum, v in enumerate(model.infer_vector(d.words)):
                x[helper, enum] = v
            y[helper] = class_map[d.tags[0]]
            helper += 1
        # y = to_categorical(y)
        data_table = shuffle(data_table)
        # print(data_table[-1][0])
        # x = np.array([x[0] for x in data_table])
        # y = np.array([class_map[y[1]] for y in data_table])
        # print(x)
        # x = pad_sequences(x, maxlen=len(x))
        # print(x)
        y = np_utils.to_categorical(y)
    #
        kf = KFold(10, shuffle=True, random_state=None)

        print("============================= 10-fold Cross-Validation training and testing ============================= \n")

        i = 1

        tableResults = []
        tableResults=[]
        # # NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(40, 10), random_state=1)
        for trainIndex, testIndex in kf.split(x):
            models = get_models()
            print(" ============== Fold ", i, "============")
            for key in models.keys():
                # try:
                print("Training " + key)
                trainDocs, testDocs = x[trainIndex], x[testIndex]
                trainCats, testCats = y[trainIndex], y[testIndex]

                models[key].fit(trainDocs, trainCats)

                scores = models[key].evaluate(testDocs, testCats, verbose=0)
                print("Accuracy: %.2f%%" % (scores[1] * 100))

                #
                # pred = models[key].predict(testDocs)
                # print(pred)
                # accuracy = accuracy_score(testCats, pred)
                # recall = recall_score(testCats, pred, average='weighted')
                # precision = precision_score(testCats, pred, average='weighted')
                # f1 = f1_score(testCats, pred, average='weighted')
                # tableResults.append({'model': key, 'accuracy': accuracy, 'recall': recall, 'precision': precision, 'f1': f1})
                # except:
                #     tableResults.append({'model': key, 'accuracy': 0, 'recall': 0, 'precision': 0, 'f1': 0})
            i+=1

        #     print('============================= Writing results from cross-validation =============================')
        #     measures = ['precision', 'recall', 'accuracy', 'f1']
        #     df = pd.DataFrame(tableResults)
        #     filt = pd.pivot_table(df, values=['precision', 'recall', 'accuracy', 'f1'], index=['model'])
        #     print(" Results")
        #     print(filt)
        #     for key in get_models().keys():
        #         with open(key+'_results/result.csv', 'a') as f:
        #             writer = csv.writer(f, delimiter=',')
        #             writer.writerow([pathname])
        #             for measure in measures:
        #                 value = [measure]
        #                 value.append(filt[measure][key])
        #                 writer.writerow(value)
        #             writer.writerow([''])
        #     # open('trained/'+pathname, 'w').close()
        '''
        Here we train the method over the doc2vec vector model for persistence
        '''
        print("============================= Training methods for persistence =============================")
        models = get_models()
        for key in models.keys():
            print("Training " + key)
            # If we already trained the method, there is no need to train it again
            if pathname + ".model" not in os.listdir('./'+key+'_saved'):
                # try:
                # print(x.shape, y.shape)
                models[key].fit(x, y)
                # joblib.dump(models[key], key+'_saved/'+pathname+'.model')
                # except:
                #     print("Model " + key +" couldn't be trained")
    keys = level.keys()
    break
    # print(keys)