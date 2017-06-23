'''
https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/test/test_doc2vec.py
https://rare-technologies.com/word2vec-tutorial/
'''

import csv
import os

import nltk
from random import shuffle

import TideneLoadCorpus
import TideneTextNormalize
import TidenePreProcess
import TideneMisc

import gensim
from gensim.models import doc2vec
from sklearn.cross_validation import *
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.metrics.classification import f1_score, classification_report, accuracy_score


def dbow(alpha, min_alpha):
    """Test DBOW doc2vec training."""
    model = doc2vec.Doc2Vec(alpha=alpha, min_alpha=min_alpha, dm=0, hs=1, negative=0,
                            min_count=2, window=2)
    return 'dbow', model

# def dm_mean(alpha, min_alpha):
# 	"""Test DM/mean doc2vec training."""
# 	model = doc2vec.Doc2Vec(dm=1, dm_mean=1, size=20, window=4, hs=1, negative=0,
# 							alpha=alpha, min_count=2, min_alpha=min_alpha)
# 	return "dm_mean", model
#
# def dm_sum(alpha, min_alpha):
# 	"""Test DM/sum doc2vec training."""
# 	model = doc2vec.Doc2Vec(dm=1, dm_mean=0, size=20, window=4, hs=1, negative=0,
# 							alpha=alpha, min_count=2, min_alpha=min_alpha)
# 	return "dm_sum", model
#
# def dm_concat(alpha, min_alpha):
# 	"""Test DM/concatenate doc2vec training."""
# 	model = doc2vec.Doc2Vec(dm=1, dm_concat=1, size=20, window=4, hs=1, negative=0,
# 							alpha=alpha, min_count=2, min_alpha=min_alpha)
# 	return "dm_concat", model
#
# def dbow_neg(alpha, min_alpha):
# 	"""Test DBOW doc2vec training."""
# 	model = doc2vec.Doc2Vec(dm=0, hs=0, negative=10, min_count=2,
# 							alpha=alpha, min_alpha=min_alpha)
# 	return "dbow_neg", model
#
# def dm_mean_neg(alpha, min_alpha):
# 	"""Test DM/mean doc2vec training."""
# 	model = doc2vec.Doc2Vec(dm=1, dm_mean=1, size=20, window=4, hs=0, negative=10,
# 							alpha=alpha, min_count=2, min_alpha=min_alpha)
# 	return "dm_mean_neg", model
#
# def dm_sum_neg(alpha, min_alpha):
# 	"""Test DM/sum doc2vec training."""
# 	model = doc2vec.Doc2Vec(dm=1, dm_mean=0, size=20, window=4, hs=0, negative=10,
# 							alpha=alpha, min_count=2, min_alpha=min_alpha)
# 	return "dm_sum_neg", model
#
# def dm_concat_neg(alpha, min_alpha):
# 	"""Test DM/concatenate doc2vec training."""
# 	model = doc2vec.Doc2Vec(dm=1, dm_concat=1, size=20, window=4, hs=0, negative=10,
# 							alpha=alpha, min_count=2, min_alpha=min_alpha)
# 	return "dm_concat_neg", model

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
            yield [cat, stream]

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
pathnames= []
for x in range(ord('D'), ord('H')+1):
    dirs = os.listdir('base5/'+chr(x))
    for directory in dirs:
        pathnames.append('base5/'+chr(x)+'/'+directory)

# pathnames = ['base', 'base2']#["base3/"+chr(x) for x in range(ord('A'),ord('H')+1) ]
print(pathnames)

for pathname in pathnames:
    print("Working on " + pathname)
    '''
    Preparing dataset
    '''
    categories = get_categories(pathname)
    print(categories)
    all_corpus = LoadCorpus(get_files_paths(pathname), tokenizer=tokenizer, stop_set=stop_set, stemmer=stemmer)
    trainDocs, testDocs = train_test_split(get_files_paths(pathname), test_size = (splitRateTestPerc/100), train_size= (splitRateTrainPerc/100), random_state = randomInt)
    trainCorpus = LoadCorpus(trainDocs, tokenizer=tokenizer, stop_set=stop_set, stemmer=stemmer)
    testCorpus = LoadCorpus(testDocs, tokenizer=tokenizer, stop_set=stop_set, stemmer=stemmer)
    '''
    Constructing the models
    '''
    print('Creating the models')
    model = dict()
    # model['dbow_no_hs'] = doc2vec.Doc2Vec(dm=0, hs=0, alpha=ALPHA, size= 20, window=4, min_alpha=MIN_ALPHA, min_count=0)
    key, m = dbow(ALPHA, MIN_ALPHA)
    model[key] = m
    # key, m = dm_mean(ALPHA, MIN_ALPHA)
    # model[key] = m
    # key, m = dm_sum(ALPHA, MIN_ALPHA)
    # model[key] = m
    # key, m = dm_concat(ALPHA, MIN_ALPHA)
    # model[key] = m
    # key, m = dbow_neg(ALPHA, MIN_ALPHA)
    # model[key] = m
    # key, m = dm_mean_neg(ALPHA, MIN_ALPHA)
    # model[key] = m
    # key, m = dm_sum_neg(ALPHA, MIN_ALPHA)
    # model[key] = m
    # key, m = dm_concat_neg(ALPHA, MIN_ALPHA)
    # model[key] = m


    for key, m in model.items():
        print("Starting training " + key)
        print("Build vocabulary")
        m.build_vocab(all_corpus)
        alpha = 0.025
        min_alpha = 0.001
        num_epochs = 20
        alpha_delta = (alpha - min_alpha) / num_epochs

        for epoch in range(num_epochs):
            print ('Now training epoch %s'%epoch)
            m.alpha = alpha
            m.min_alpha = alpha  # fix the learning rate, no decay
            m.train(trainCorpus)
            alpha -= alpha_delta
        print("Saving model")
        m.save(pathname.replace('/', '_')+".model")
        actual = []
        pred = []
        for doc in testCorpus:
            actual.append(doc.tags)
            predVec = m.infer_vector(doc.words)
            predTags = m.docvecs.most_similar([predVec], topn=5)
            # print(predTags)
            pred.append(predTags[0][0])


        document_matrix = confusion_matrix(actual, pred, labels=categories)
        document_recall = recall_score(actual, pred, average='weighted')
        document_precision = precision_score(actual, pred, average='weighted')
        document_f1 = f1_score(actual, pred, average='weighted')
        document_accuracy = accuracy_score(actual, pred)
        report = classification_report(actual, pred, labels=categories)
        # with open(pathname.replace('/', '_')+'_report', 'w') as file:
        # 	file.write(report)
        with open(pathname.replace('/', '_')+'.csv', 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            # writer.writerow(categories)
            # for row in document_matrix:
            # 	writer.writerow(row)
            writer.writerow(['Precision', document_precision])
            writer.writerow(['Recall', document_recall])
            writer.writerow(['Accuracy', document_accuracy])
            writer.writerow(['F1', document_f1])

    # model = doc2vec.Doc2Vec.load("dbow.model")
    # print(model.docvecs[0])