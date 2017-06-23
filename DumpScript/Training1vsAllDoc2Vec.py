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
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.linear_model.stochastic_gradient import SGDClassifier
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


def dbow(alpha, min_alpha):
    """Test DBOW doc2vec training."""
    model = doc2vec.Doc2Vec(alpha=alpha, min_alpha=min_alpha, dm=0, hs=1, negative=0,
                            min_count=2, window=2, workers=4)
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
                if key == self.key:
                    yield ['True', stream]
                else:
                    yield ['False', stream]

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

def get_models():
    models = dict()
    models['NN'] = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(40, 10), random_state=1)
    models['sgdc'] = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)
    return models

def create_directories(keys):
    if 'doc2vec_saved' not in os.listdir('./'):
        os.mkdir('doc2vec_saved')
    for key in keys:
        if key+'_saved' not in os.listdir('./'):
            os.mkdir(key+'_saved')
        if key+'_results' not in os.listdir('./'):
            os.mkdir(key+'_results')
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
# Get files absolute path from directory
path = 'base5'
create_directories(get_models().keys())
for level in HierarchicalStructure(get_files_paths(path)):
    for key in level:
        pathname = key
        print("Working on " + pathname)
        # if pathname +".model" not in os.listdir('./doc2vec_saved'):
        #     print("============================= Training doc2vec for " + pathname + " =============================")
        #     '''
        #     Preparing dataset
        #     '''
        #     all_corpus = LoadCorpus(key, level, tokenizer=tokenizer, stop_set=stop_set, stemmer=stemmer)
        #     '''
        #     Constructing the models
        #     '''
        #     print('Creating the models')
        #     model = dict()
        #     key, m = dbow(ALPHA, MIN_ALPHA)
        #     model[key] = m
        #     '''
        #     Trainig doc2vec
        #     '''
        #     for key, m in model.items():
        #         print("Starting training " + key)
        #         print("Build vocabulary")
        #         m.build_vocab(all_corpus)
        #         alpha = 0.025
        #         min_alpha = 0.001
        #         num_epochs = 10
        #         alpha_delta = (alpha - min_alpha) / num_epochs
        #
        #         for epoch in range(num_epochs):
        #             print ('Now training epoch %s'%epoch)
        #             m.alpha = alpha
        #             m.min_alpha = alpha  # fix the learning rate, no decay
        #             m.train(all_corpus)
        #             alpha -= alpha_delta
        #         print("Saving model")
        #         m.save("doc2vec_saved/"+pathname.split('/')[-1]+".model")


        if pathname not in os.listdir('./trained'):
            print("Getting vectors for " + pathname)
            model = doc2vec.Doc2Vec.load("doc2vec_saved/"+pathname.split('/')[-1]+".model")
            files_paths = get_files_paths(pathname)
            data_table = []
            classes_list = []
            data = LoadCorpus(key, level, tokenizer=tokenizer, stop_set=stop_set, stemmer=stemmer)
            for d in data:
                data_table.append([model.infer_vector(d.words), d.tags[0]])
            data_table = shuffle(data_table)
            x = np.array([x[0] for x in data_table])
            y = np.array([y[1] for y in data_table])

            kf = KFold(10, shuffle=True, random_state=None)

            print("============================= 10-fold Cross-Validation training and testing ============================= \n")

            i = 1

            tableResults = []
            tableResults=[]
            # NN = LogisticRegression(n_jobs=1, C=1e5)
            for trainIndex, testIndex in kf.split(x):
                models = get_models()
                print(" ============== Fold ", i, "============")
                for key in models.keys():
                    try:
                        print("Training " + key)
                        trainDocs, testDocs = x[trainIndex], x[testIndex]
                        trainCats, testCats = y[trainIndex], y[testIndex]
                        models[key].fit(trainDocs, trainCats)
                        pred = models[key].predict(testDocs)
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
            filt = pd.pivot_table(df, values=['precision', 'recall', 'accuracy', 'f1'], index=['model'])
            print(" Results")
            print(filt)
            for key in get_models().keys():
                with open(key + '_results/result.csv', 'a') as f:
                    writer = csv.writer(f, delimiter=',')
                    writer.writerow([pathname])
                    for measure in measures:
                        value = [measure]
                        value.append(filt[measure][key])
                        writer.writerow(value)
                    writer.writerow([''])
            open('trained/' + pathname, 'w').close()

            '''
            Here we train the method over the doc2vec vector model for persistence
            '''
            print("============================= Training methods for persistence =============================")
            models = get_models()
            for key in models.keys():
                print("Training " + key)
                # If we already trained the method, there is no need to train it again
                if pathname + ".model" not in os.listdir('./' + key + '_saved'):
                    try:
                        models[key].fit(x, y)
                        joblib.dump(models[key], key + '_saved/' + pathname + '.model')
                    except:
                        print("Model " + key + " couldn't be trained")