import gensim
import nltk
from DeepLearning import database, learn
from DeepLearning.helper import dictionary_to_list
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

word2vec_trainer = learn.Word2VecTrainer()
word2vec_model = word2vec_trainer.load_model('word2vec2.model')


some_vector = word2vec_model.wv['big'] - word2vec_model.wv['biggest'] + word2vec_model.wv['small']
print(word2vec_model.wv.most_similar(positive=['big', 'small'], negative=['biggest']))

