#def load_src(name, fpath):
#    import os, imp
#    return imp.load_source(name, os.path.join(os.path.dirname(__file__), fpath))

#load_src("database", "../DeepLearning/database.py")
#load_src("helper","../DeepLearning/helper.py")



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

output_model_file = '../doc2vec_models/doc2vec_old_400.model'
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
documents = mongodb.get_all_meta('word2vec_old')
all_corpus = MongoLoadDocumentData('patents', documents, clean_text=True, tokenizer=tokenizer, stop_set=stop_set,
                                   abstract=True, description=True, doc2vec_doc=True)
size = 400
iteration = 30

# print("============================= Loading data =============================")
# data_dict = database.FlatStructureDatabase('../../database/descriptions/base').subclasses()
# data_vec = dictionary_to_list(data_dict)
# all_corpus = database.LoadFilesContent(data_vec, tokenizer=tokenizer, stop_set=stop_set)
#
# for c in all_corpus:
#     print(c)

if new_model:
    print("=============================== Training Model ===============================")

    doc2vecTrainer = learn.Doc2VecTrainer(iter=iteration, min_alpha=0.0001, size=size)
    doc2vecTrainer.train(all_corpus)
# else:
#     print("=============================== Training Model ===============================")
#     word2vecTrainer = learn.Word2VecTrainer(iter=iteration, size=size)
#     model = word2vecTrainer.load_model(input_model_file)
#     word2vecTrainer.retrain(model, all_corpus, sg=sg)
#
#     doc2vecTrainer = learn.Doc2VecTrainer(iter=20, min_alpha=0.0001)
#     doc2vecTrainer.train(all_corpus)

doc2vecTrainer.save(output_model_file)
