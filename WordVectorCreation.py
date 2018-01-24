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
collection = 'training_docs100'
documents = mongodb.get_all_meta(collection)
corpus = MongoLoadDocumentData('patents', documents, clean_text=True, tokenizer=tokenizer, stop_set=stop_set,description=True)

word2vec_model = dl.learn.Word2VecTrainer().load_model('../word2vec_models/word2vec_400.model')
word_vector_generator = dl.data_representation.Word2VecEmbeddingCreator(word2vec_model, maxWords=150, embeddingSize=400)

shuffled = []
for document in documents:
    shuffled.append(document["filename"])
shuffle(shuffled)

i=0
for doc in shuffled:
    document = mongodb.get_document_by(collection, 'filename', doc)
    # print(document['filename'])
    if i%1000 == 0:
        print(str(i) + ' ' + document['filename'])
    content = corpus.get_file_content(document['filename'])
    content = corpus.clean(content['description'])
    word_embedding_matrix = word_vector_generator.create_x_text(content)
    client = pymongo.MongoClient()
    patents_database = client.patents
    word_embedding_collection = patents_database.training_embedding_old_400
    document['embedding'] = bson.binary.Binary(pickle.dumps(word_embedding_matrix, protocol=2))
    word_embedding_collection.insert_one(document)
    i+=1
    # fs = gridfs.GridFS(patents_database, collection="documents_embedding_docs100")
    # with fs.new_file(filename=document['filename'], content_type="binary") as fp:
    #     fp.write(word_embedding_matrix)
    # print(content)

