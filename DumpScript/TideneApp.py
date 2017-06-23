'''
Version 15-12-2016
Python 3.4.3
'''
from __future__ import division
import nltk
import gensim
import re
import os
#import numpy as np

from gensim.models.doc2vec import LabeledSentence
from sklearn.cross_validation import *
from os import path
from random import shuffle


import TideneLoadCorpus
import TideneTextNormalize
import TidenePreProcess
import TideneMisc

'''
     Working path
'''
pathname = "PatentsToy"

# usage: python TideneApp.py



'''
	Configs
'''
language = 'portuguese'
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
stopSet = nltk.corpus.stopwords.words(language)
stemmer = nltk.stem.RSLPStemmer()


'''
Loading Patents
'''
corpus = TideneTextNormalize.TextNormalizePort(TideneLoadCorpus.GetCorpusFromPath(pathname)) # corpus = [[category,text],[category,text]]
#for text in corpus:
#	print(text)


'''
Structuring patents on a dataframe
'''
dataframe = TideneMisc.LstToDataFrame.transform(corpus)  # return a data frame with data and target/cats
catNames = list(set(dataframe.target))   # categories/classes
#print("CatNames:",catNames)
#print (dataframe.data) # 0  azo corantes escarlates reativos a fibra azo c...
#print (dataframe.target) # 0     C07D


# preprocess docs and fils LabeledSenceStructure
dataset = []
for index,doc in enumerate(dataframe.data):
    preprocDoc = TidenePreProcess.CleanStopWords(stopSet,TidenePreProcess.TokenizeFromList(tokenizer,[doc])) #returns a list
    for p in preprocDoc:
        dataset.append(LabeledSentence(words=p, tags=[dataframe.target[index]]))  # tags = list


# split modelo into training and tes
print (" Gensim 80-20 training and testing \n")
splitRateTrainPerc = 80
splitRateTestPerc = 20
randomInt = 42
trainDocs,testDocs = train_test_split(dataset, test_size = (splitRateTestPerc/100), train_size= (splitRateTrainPerc/100), random_state = randomInt)

# ========================== training ============================
# http://sujitpal.blogspot.com.br/2016/04/predicting-movie-tags-from-plots-using.html

print("================= Building model =============")
model = gensim.models.Doc2Vec(dm = 0, alpha=0.025, size= 20, window=4, min_alpha=0.025, min_count=0)

model.build_vocab(trainDocs)

print("================ Starting training ===========")
alpha = 0.025
min_alpha = 0.001
num_epochs = 20
alpha_delta = (alpha - min_alpha) / num_epochs

for epoch in range(num_epochs):
    #print ('Now training epoch %s'%epoch)
    shuffle(trainDocs)
    model.alpha = alpha
    model.min_alpha = alpha  # fix the learning rate, no decay
    model.train(trainDocs)
    alpha -= alpha_delta

model.save('mymodeldoc2vec')

# ==================================== testing ============================

model = gensim.models.Doc2Vec.load('mymodeldoc2vec')

# evaluate the model
tot_sim = 0.0
pred = []
for doc in testDocs:
    print("=============== doc ========== ")
    print(doc)
    print("====== actual tag ========= ")
    print(doc.tags)
    predVec = model.infer_vector(doc.words)
    print("===== pred sim tags ==========")
    predTags = model.docvecs.most_similar([predVec], topn=5)
    for pt in predTags:
        print(pt)
    print("=============================")
