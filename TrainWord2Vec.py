import gensim
import nltk
from DeepLearning import database, learn
from DeepLearning.helper import dictionary_to_list
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

'''
Configurations
'''
language = 'english'
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
stop_set = nltk.corpus.stopwords.words(language)
stemmer = gensim.parsing.PorterStemmer()

print("============================= Training word2vec =============================")
data_dict = database.FlatStructureDatabase('../database/descriptions/base').subclasses()
data_vec = dictionary_to_list(data_dict)
all_corpus = database.LoadFilesContent(data_vec, tokenizer=tokenizer, stop_set=stop_set)

word2vecTrainer = learn.Word2VecTrainer(iter=15, size=40)
word2vecTrainer.train(all_corpus)
word2vecTrainer.save('word2vec_model.model')