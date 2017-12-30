import gensim
import nltk
from DeepLearning import database, learn
from DeepLearning.helper import dictionary_to_list
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

word2vec_trainer = learn.Word2VecTrainer()
word2vec_model = word2vec_trainer.load_model('word2vec.model')
word2vec_model2 = word2vec_trainer.load_model('word2vec_mongo.model')

# question = open('../TrainedLSTM/question-words.txt')
len(word2vec_model.wv.vocab)
#
# some_vector = word2vec_model.wv['big'] - word2vec_model.wv['biggest'] + word2vec_model.wv['small']
# print(word2vec_model.wv.most_similar(positive=['big', 'small'], negative=['biggest']))

print("===================================== First Model ==========================================")
accuracy = word2vec_model.wv.accuracy('../TrainedLSTM/question-words.txt')

sum_corr = len(accuracy[-1]['correct'])
sum_incorr = len(accuracy[-1]['incorrect'])
total = sum_corr + sum_incorr
percent = lambda a: a / total * 100


print('Total sentences: {}\nCorrect: {:.2f}%\n Incorrect: {:.2f}%\n'.format(total, percent(sum_corr), percent(sum_incorr)))

print('Vocabulary Length: {}'.format(len(word2vec_model.wv.vocab)))
vocab1 = set(word2vec_model.wv.vocab.keys())

print("===================================== Second Model ==========================================")
accuracy = word2vec_model2.wv.accuracy('../TrainedLSTM/question-words.txt')

sum_corr = len(accuracy[-1]['correct'])
sum_incorr = len(accuracy[-1]['incorrect'])
total = sum_corr + sum_incorr
percent = lambda a: a / total * 100


print('Total sentences: {}\nCorrect: {:.2f}%\n Incorrect: {:.2f}%\n'.format(total, percent(sum_corr), percent(sum_incorr)))

print('Vocabulary Length: {}'.format(len(word2vec_model2.wv.vocab)))
vocab2 = set(word2vec_model2.wv.vocab.keys())

dif = vocab1.difference(vocab2)
