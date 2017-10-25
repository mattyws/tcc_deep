from DeepLearning.database import MongoLoadDocumentMeta, MongoLoadDocumentData
import gensim
import nltk
from DeepLearning import database, learn
from DeepLearning.helper import dictionary_to_list
import logging
import getopt
import sys

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sg = 1

try:
    opts, args = getopt.getopt(sys.argv[1:], "ho:m:")
except getopt.GetoptError:
    print('test.py -m <model_file> -t')
    sys.exit(2)

output_model_file = 'word2vec_mongo.model'
new_model = True
retrain = False
input_model_file = ''
for opt, arg in opts:
    if opt == '-h':
        print('test.py -m <model_file> -t')
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
mongodb = MongoLoadDocumentMeta('patents')
documents = mongodb.get_all_meta('word2vec_docs')
all_corpus = MongoLoadDocumentData('patents', documents, clean_text=True, tokenizer=tokenizer, stop_set=stop_set, abstract=True, description=True, claims=True)

# print("============================= Loading data =============================")
# data_dict = database.FlatStructureDatabase('../../database/descriptions/base').subclasses()
# data_vec = dictionary_to_list(data_dict)
# all_corpus = database.LoadFilesContent(data_vec, tokenizer=tokenizer, stop_set=stop_set)
#
# for c in all_corpus:
#     print(c)

if new_model:
    print("=============================== Training Model ===============================")
    word2vecTrainer = learn.Word2VecTrainer(iter=15, size=200)
    word2vecTrainer.train(all_corpus, sg=sg)
else:
    print("=============================== Training Model ===============================")
    word2vecTrainer = learn.Word2VecTrainer(iter=2, size=40)
    model = word2vecTrainer.load_model(input_model_file)
    word2vecTrainer.retrain(model, all_corpus, sg=sg)

word2vecTrainer.save(output_model_file)