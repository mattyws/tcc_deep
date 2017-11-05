import gensim
import nltk
import logging

from DeepLearning import database, learn
from DeepLearning.database import MongoLoadDocumentMeta, MongoLoadDocumentData


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sg = 1

'''
Configurations
'''
language = 'english'
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
stop_set = nltk.corpus.stopwords.words(language)
stemmer = gensim.parsing.PorterStemmer()
mongodb = MongoLoadDocumentMeta('patents')
documents = mongodb.get_all_meta('doc2vec_docs')
all_corpus = MongoLoadDocumentData('patents', documents, clean_text=True, tokenizer=tokenizer, stop_set=stop_set, abstract=True, description=True, claims=True, doc2vec_doc=True)

# for c in all_corpus:
#     print(c)

doc2vecTrainer = learn.Doc2VecTrainer(iter=20, min_alpha=0.0001)
doc2vecTrainer.train(all_corpus)
doc2vecTrainer.save('doc2vec_mongo.model')