from random import shuffle

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
collection = 'testing_docs100'
documents = mongodb.get_all_meta(collection)
corpus = MongoLoadDocumentData('patents', documents, clean_text=True, tokenizer=tokenizer, stop_set=stop_set,description=True, doc2vec_doc=True)

doc2vec_model = dl.learn.Doc2VecTrainer().load_model('../doc2vec_models/doc2vec_old_50_dif.model')
doc_vector_generator = dl.data_representation.Doc2VecEmbeddingCreator(doc2vec_model)
shuffled = []
for document in documents:
    shuffled.append(document["filename"])
shuffle(shuffled)

i = 0
for doc in shuffled:
    document = mongodb.get_document_by(collection, 'filename', doc)
    if i%1000 == 0:
        print(str(i) + ' ' + document['filename'])
    content = corpus.get_file_content(document['filename'])
    content = corpus.clean(content['description'])
    doc_embedding_vector = doc_vector_generator.create_x_text(content).reshape((1,50))
    client = pymongo.MongoClient()
    patents_database = client.patents
    doc_embedding_collection = patents_database.testing_document_embedding_old_50_dif
    document['embedding'] = bson.binary.Binary(pickle.dumps(doc_embedding_vector, protocol=2))
    doc_embedding_collection.insert_one(document)
    i+=1
