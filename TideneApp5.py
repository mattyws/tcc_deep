'''

Training doc2vec separating each document as a diferent labels
'''

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


class TideneGensimDoc2VecVectorizer(object):
    # inspired in https://github.com/linanqiu/word2vec-sentiments/blob/master/word2vec-sentiment.ipynb

    def __init__(self, doc2vec, size):
        self.doc2vec = doc2vec
        self.array = None
        self.size = size

    def fit(self, X, y):
        # transforming to doc2vec arrays
        self.array = np.zeros((len(X), self.size))
        idx = 0
        for tag in y:
            self.array[idx] = self.doc2vec.docvecs[tag]
            idx += 1
        return self

    def transform(self, X):
        # infering a new vector
        array = np.zeros((len(X), self.size))
        idx = 0
        for d in X:
            array[idx] = self.doc2vec.infer_vector(d)
            idx += 1
        return array

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

class GetFilesFromPath(object):
    def __init__(self, path_list):
        self.path_list = path_list
    def __iter__(self):
        shuffle(self.path_list)
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
                yield doc2vec.TaggedDocument(t, [data[0]])
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
pathnames= ['base']+["base3/"+chr(x) for x in range(ord('A'),ord('H')+1)]
for x in range(ord('A'), ord('H')+1):
    dirs = os.listdir('base5/'+chr(x))
    for directory in dirs:
        pathnames.append('base5/'+chr(x)+'/'+directory)
# pathnames = ['base']#["base3/"+chr(x) for x in range(ord('A'),ord('H')+1) ]
print(pathnames)
pathnames.sort()

for pathname in pathnames:
    print("Working on " + pathname)
    categories = get_categories(pathname)
    if pathname.split('/')[-1]+".model" not in os.listdir('./doc2vec_saved'):
        print("Training doc2vec for " + pathname)
        '''
        Preparing dataset
        '''
        categories = get_categories(pathname)
        print(categories)
        all_corpus = LoadCorpus(get_files_paths(pathname), tokenizer=tokenizer, stop_set=stop_set, stemmer=stemmer)
        # trainDocs, testDocs = train_test_split(get_files_paths(pathname), test_size = (splitRateTestPerc/100), train_size= (splitRateTrainPerc/100), random_state = randomInt)
        # trainCorpus = LoadCorpus(trainDocs, tokenizer=tokenizer, stop_set=stop_set, stemmer=stemmer)
        # testCorpus = LoadCorpus(testDocs, tokenizer=tokenizer, stop_set=stop_set, stemmer=stemmer)
        '''
        Constructing the models
        '''
        print('Creating the models')
        model = dict()
        key, m = dbow(ALPHA, MIN_ALPHA)
        model[key] = m
        '''
        Trainig doc2vec model
        '''
        for key, m in model.items():
            print("Starting training " + key)
            print("Build vocabulary")
            m.build_vocab(all_corpus)
            alpha = 0.025
            min_alpha = 0.001
            num_epochs = 10
            alpha_delta = (alpha - min_alpha) / num_epochs

            for epoch in range(num_epochs):
                print ('Now training epoch %s'%epoch)
                m.alpha = alpha
                m.min_alpha = alpha  # fix the learning rate, no decay
                m.train(all_corpus)
                alpha -= alpha_delta
            print("Saving model")
            m.save("doc2vec_saved/"+pathname.split('/')[-1]+".model")
    print("Preparing data for neural network for " + pathname)
    model = doc2vec.Doc2Vec.load("doc2vec_saved/"+pathname.split('/')[-1]+".model")
    data_table = []
    for i in range(0, len(model.docvecs)):
        key = model.docvecs.index_to_doctag(i)
        data_table.append([model.docvecs[key], key])
        # print(model.docvecs.indexed_doctags(model.docvecs[key]))
        # print(model.docvecs[key], key.split('_')[0])

    x = np.array([x[0] for x in data_table])
    y = np.array([y[1] for y in data_table])

    kf = KFold(10, shuffle=True, random_state=None)

    print(" 10-fold Cross-Validation training and testing \n")

    i = 1

    tableResults = []
    tableResults=[]
    NN = MLPClassifier(solver='lbfgs', alpha=1e-8, hidden_layer_sizes=(40, 10), random_state=1)
    for trainIndex, testIndex in kf.split(x):
        print(" ============== Fold ", i, "============\n")
        trainDocs, testDocs = x[trainIndex], x[testIndex]
        trainCats, testCats = y[trainIndex], y[testIndex]
        NN.fit(trainDocs, trainCats)
        pred = NN.predict(testDocs)
        accuracy = accuracy_score(testCats, pred)
        recall = recall_score(testCats, pred, average='weighted')
        precision = precision_score(testCats, pred, average='weighted')
        f1 = f1_score(testCats, pred, average='weighted')
        tableResults.append({'model': 'NN', 'accuracy': accuracy, 'recall': recall, 'precision': precision, 'f1': f1})
        i+=1
    joblib.dump(NN, 'NN_saved/' + pathname.split('/')[-1]+'.model')
    NN = joblib.load('NN_saved/' + pathname.split('/')[-1]+'.model')
    measures = ['precision', 'recall', 'accuracy', 'f1']
    with open('neuralNN' + pathname.replace('/', '_')+'.csv', 'w') as f:
        writer = csv.writer(f, delimiter=',')
        df = pd.DataFrame(tableResults)
        filt = pd.pivot_table(df, values=['precision', 'recall', 'accuracy', 'f1'], index=['model'])
        print(" Results")
        print(filt)
        for measure in measures:
            value = [measure]
            value.append(filt[measure]['NN'])
            writer.writerow(value)
