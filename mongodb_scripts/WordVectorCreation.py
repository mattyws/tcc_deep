import bson
import gensim
import nltk
import pickle
import pymongo
import gridfs
import DeepLearning as dl
from DeepLearning.database import MongoLoadDocumentMeta, MongoLoadDocumentData
import sys

def addToDatabase(patent_structure):
    client = pymongo.MongoClient()
    patents_database = client.patents
    #Using diferent collections in gridfs
    description_fs = gridfs.GridFS(patents_database, collection="descriptions")
    abstract_fs = gridfs.GridFS(patents_database, collection="abstracts")
    claims_fs = gridfs.GridFS(patents_database, collection="claims")

    if not description_fs.exists(filename=patent_structure["fname"]):
        #Creating metadata
        docs_meta = patents_database.documents_meta
        docs_meta.insert_one({
            'filename' : patent_structure['fname'],
            'ipc_classes' : patent_structure['classes']
        })
        # Writting files
        with description_fs.new_file(filename=patent_structure['fname'],content_type="plain/text") as fp:
            fp.write(patent_structure['description'].encode('UTF-8'))
        with abstract_fs.new_file(filename=patent_structure['fname'],content_type="plain/text") as fp:
            fp.write(patent_structure['claims'].encode('UTF-8'))
        with claims_fs.new_file(filename=patent_structure['fname'],content_type="plain/text") as fp:
            fp.write(patent_structure['abstract'].encode('UTF-8'))

'''
Configurations
'''
language = 'english'
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
stop_set = nltk.corpus.stopwords.words(language)
stemmer = gensim.parsing.PorterStemmer()
mongodb = MongoLoadDocumentMeta('patents')
documents = mongodb.get_all_meta('training_docs100')
corpus = MongoLoadDocumentData('patents', documents, clean_text=True, tokenizer=tokenizer, stop_set=stop_set,description=True)

word2vec_model = dl.learn.Word2VecTrainer().load_model('../word2vec.model')
word_vector_generator = dl.data_representation.Word2VecEmbeddingCreator(word2vec_model, maxWords=200, embeddingSize=200)

for document in documents:
    print(document['filename'])
    content = corpus.get_file_content(document['filename'])
    content = corpus.clean(content['description'])
    word_embedding_matrix = word_vector_generator.create_x_text(content)
    client = pymongo.MongoClient()
    patents_database = client.patents
    word_embedding_collection = patents_database.documents_embedding_docs100
    document['embedding'] = bson.binary.Binary(pickle.dumps(word_embedding_matrix, protocol=2))
    word_embedding_collection.insert_one(document)
    # fs = gridfs.GridFS(patents_database, collection="documents_embedding_docs100")
    # with fs.new_file(filename=document['filename'], content_type="binary") as fp:
    #     fp.write(word_embedding_matrix)
    # print(content)

