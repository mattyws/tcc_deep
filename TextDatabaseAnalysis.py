import os
from random import shuffle

import gensim
import nltk

import DeepLearning as dl
import numpy as np

from DeepLearning import database
from DeepLearning.helper import *

timer = TimerCounter()
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
stop_set = nltk.corpus.stopwords.words('english')
stemmer = gensim.parsing.PorterStemmer()

# Getting the hierarchcal structures from the database, and looping over it
data = dl.database.FlatStructureDatabase('../database/descriptions/base').subclasses()

lower_class_documents = 1200
class_lower_document = None
for classes in data.keys():
    print("Class " + classes + " has " + str(len(data[classes])) + " documents")
    if lower_class_documents > len(data[classes]):
        lower_class_documents = len(data[classes])
        class_lower_document = classes
print("The class " + class_lower_document + " has the lowest number of documents with the value : " + str(lower_class_documents))


# data_vec = dictionary_to_list(data)
# all_corpus = database.LoadFilesContent(data_vec, tokenizer=tokenizer)
# print("Data len: " + str(len(all_corpus)))
# total_words = 0
# total_words_no_stop = 0
# unique_tokens = dict()
# for text in all_corpus:
#     for word in text:
#         total_words += 1
#         if word not in stop_set:
#             total_words_no_stop += 1
#         if word not in unique_tokens.keys():
#             unique_tokens[word] = 0
#         unique_tokens[word] += 1
#
# print("Total words in data: " + str(total_words))
# print("Total words with no stop words: " +str(total_words_no_stop))
# print("Total unique tokens: " + str(len(unique_tokens.keys())))

