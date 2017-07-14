import gensim
import nltk
from DeepLearning import database, learn
from DeepLearning.helper import dictionary_to_list
import logging
import getopt
import sys

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


try:
  opts, args = getopt.getopt(sys.argv[1:], "ho:m:")
except getopt.GetoptError:
  print ('test.py -m <model_file> -t')
  sys.exit(2)

output_model_file = 'word2vec.model'
new_model = True
retrain = False
input_model_file = ''
for opt, arg in opts:
    if opt == '-h':
        print ('test.py -m <model_file> -t')
        sys.exit()
    elif opt in ("-m", "--model"):
        input_model_file = arg
        new_model = False
    elif opt in ("-o", "--output"):
        output_model_file = arg

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


if new_model:
    print("=============================== Training Model ===============================")
    word2vecTrainer = learn.Word2VecTrainer(iter=15, size=40)
    word2vecTrainer.train(all_corpus)
else:
    print("=============================== Training Model ===============================")
    word2vecTrainer = learn.Word2VecTrainer(iter=15, size=40)
    model = word2vecTrainer.load_model(input_model_file)
    word2vecTrainer.retrain(model, all_corpus)

word2vecTrainer.save(output_model_file)