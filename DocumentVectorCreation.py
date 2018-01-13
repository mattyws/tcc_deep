import bson
import gensim
import nltk
import pickle
import pymongo
import gridfs
import DeepLearning as dl
from DeepLearning.database import MongoLoadDocumentMeta, MongoLoadDocumentData
import sys

'''
Configurations
'''
language = 'english'
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
stop_set = nltk.corpus.stopwords.words(language)
stemmer = gensim.parsing.PorterStemmer()
mongodb = MongoLoadDocumentMeta('patents')
documents = mongodb.get_all_meta('testing_docs100')
corpus = MongoLoadDocumentData('patents', documents, clean_text=True, tokenizer=tokenizer, stop_set=stop_set,description=True, doc2vec_doc=True)

doc2vec_model = dl.learn.Doc2VecTrainer().load_model('../doc2vec_models/doc2vec_old.model')
doc_vector_generator = dl.data_representation.Doc2VecEmbeddingCreator(doc2vec_model)


for doc in corpus:
    vector = doc_vector_generator.create_x(doc)
    print(vector)
