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

i = 0
for document in documents:
    if i%1000 == 0:
        print(str(i) + ' ' + document['filename'])
    content = corpus.get_file_content(document['filename'])
    content = corpus.clean(content['description'])
    doc_embedding_vector = doc_vector_generator.create_x_text(content)
    print(doc_embedding_vector)
    # client = pymongo.MongoClient()
    # patents_database = client.patents
    # word_embedding_collection = patents_database.testing_embedding_400
    # document['embedding'] = bson.binary.Binary(pickle.dumps(word_embedding_matrix, protocol=2))
    # word_embedding_collection.insert_one(document)
    i+=1
