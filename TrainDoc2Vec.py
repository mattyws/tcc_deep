import gensim
import nltk
from DeepLearning import database, learn

'''
Configurations
'''
language = 'english'
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
stop_set = nltk.corpus.stopwords.words(language)
stemmer = gensim.parsing.PorterStemmer()

print("============================= Training word2vec =============================")

all_corpus = database.LoadTextCorpus(database.FlatStructureDatabase('../database/descriptions/base').subclasses(), tokenizer=tokenizer, stop_set=stop_set)

doc2vecTrainer = learn.Doc2VecTrainer(iter=250, min_alpha=0.0001)
doc2vecTrainer.train(all_corpus)
doc2vecTrainer.save('doc2vec.model')