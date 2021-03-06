'''
A tfidf approach
'''
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.svm.classes import SVC

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
            if not path.split('/')[-4] in section_dict.keys():
                section_dict[path.split('/')[-4]] = []
            section_dict[path.split('/')[-4]].append(path)
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

def get_pipeline_models():
    pipelineModels = dict()
    pipelineModels['NN2'] = Pipeline([("vectorizer", TfidfVectorizer()), ("classifier", MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(40, 10), random_state=1))])
    # pipelineModels['nb'] = Pipeline([("vectorizer", TfidfVectorizer()), ("classifier", MultinomialNB())])
    # pipelineModels['svm'] = Pipeline([("vectorizer", TfidfVectorizer()), ("classifier", SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))])
    # # pipelineModels['svc'] = Pipeline([("vectorizer", TfidfVectorizer()), ("classifier", SVC(kernel="linear"))])
    # pipelineModels['logreg'] = Pipeline([("vectorizer", TfidfVectorizer()), ("classifier", LogisticRegression(n_jobs=1, C=1e5))])
    # pipelineModels['nc'] = Pipeline([("vectorizer", TfidfVectorizer()), ("classifier", NearestCentroid())])
    return pipelineModels
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
structure = HierarchicalStructure(get_files_paths('base5'))
keys = None
for level in structure:
    #If keys is not None, so we already train for the section level
    if keys:
        data_paths = separate_class(keys, level)
    else:
        data_paths = [level]
    #Looping through the files
    for dataP in data_paths:
        pathname = get_pathname(keys, dataP)
        if pathname not in os.listdir('trained'):
            print("Working on " + pathname)
            print("============================= Preparing data ============================= \n")
            '''
            Preparing dataset
            '''
            all_corpus = LoadCorpus(dataP, tokenizer=tokenizer, stop_set=stop_set, stemmer=stemmer)
            x = np.array([' '.join(x[0]) for x in all_corpus if len(x[0]) > 0])
            y = np.array([y[1] for y in all_corpus])
            for p in x:
                print(p)
                break
            kf = KFold(10, shuffle=True, random_state=None)

            print("============================= 10-fold Cross-Validation training and testing ============================= \n")

            i = 1

            tableResults = []
            for key in get_pipeline_models().keys():
                if not os.path.exists(key+'_saved'):
                    os.mkdir(key+'_saved')
                if not os.path.exists(key+'_results'):
                    os.mkdir(key+'_results')
                if not os.path.exists(key+'_tfidf'):
                    os.mkdir(key+'_tfidf')
            # Training all models
            for trainIndex, testIndex in kf.split(x):
                print(" ============== Fold ", i, "============\n")
                pipelineModels = get_pipeline_models()
                for key in pipelineModels.keys():
                    if not os.path.exists(key+'_saved/' + pathname.split('/')[-1]+'.model'):
                        try:
                            print("Training " + key)
                            trainDocs, testDocs = x[trainIndex], x[testIndex]
                            trainCats, testCats = y[trainIndex], y[testIndex]
                            pipelineModels[key].fit(trainDocs, trainCats.ravel())
                            pred = pipelineModels[key].predict(testDocs)
                            accuracy = accuracy_score(testCats, pred)
                            recall = recall_score(testCats, pred, average='weighted')
                            precision = precision_score(testCats, pred, average='weighted')
                            f1 = f1_score(testCats, pred, average='weighted')
                            tableResults.append({'model': key, 'accuracy': accuracy, 'recall': recall, 'precision': precision, 'f1': f1})
                        except:
                            tableResults.append({'model': key, 'accuracy': 0, 'recall': 0, 'precision': 0, 'f1': 0})
                i+=1


            print('============================= Writing results from cross-validation =============================')
            measures = ['precision', 'recall', 'accuracy', 'f1']
            df = pd.DataFrame(tableResults)
            filt = pd.pivot_table(df, values=measures, index=['model'])
            print(" Results")
            print(filt)
            for key in get_pipeline_models().keys():
                with open(key + '_results/result.csv', 'a') as f:
                    writer = csv.writer(f, delimiter=',')
                    writer.writerow([pathname])
                    for measure in measures:
                        value = [measure]
                        value.append(filt[measure][key])
                        writer.writerow(value)
                    writer.writerow([''])
            open('trained/' + pathname, 'w').close()

        pipelineModels = get_pipeline_models()
        print("============================= Training methods for persistence =============================")
        for key in pipelineModels.keys():
            if pathname + ".model" not in os.listdir('./' + key + '_saved'):
                try:
                    print("Training " + key)
                    pipelineModels[key].fit(x,y.ravel())
                    joblib.dump(pipelineModels[key].named_steps['classifier'], key + '_saved/' + pathname.split('/')[-1] + '.model')
                    with open(key + '_tfidf/' + pathname.split('/')[-1] + '.model', 'wb') as f:
                        joblib.dump(pipelineModels[key].named_steps['vectorizer'], f)
                except:
                    print("Model " + key + " couldn't be trained")


    keys = level.keys()