import gensim
import nltk
from DeepLearning import database, learn
from DeepLearning.helper import dictionary_to_list
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

word2vec_trainer = learn.Word2VecTrainer()
word2vec_model = word2vec_trainer.load_model('../word2vec_models/word2vec_400.model')

# question = open('../TrainedLSTM/question-words.txt')


some_vector = word2vec_model.wv['big'] - word2vec_model.wv['biggest'] + word2vec_model.wv['small']
print(word2vec_model.wv.most_similar(positive=['big', 'small'], negative=['biggest']))
accuracy = word2vec_model.wv.accuracy('../TrainedLSTM/question-words.txt')

sum_corr = len(accuracy[-1]['correct'])
sum_incorr = len(accuracy[-1]['incorrect'])
total = sum_corr + sum_incorr
percent = lambda a: a / total * 100

print('Total sentences: {}, Correct: {:.2f}%, Incorrect: {:.2f}%'.format(total, percent(sum_corr), percent(sum_incorr)))

