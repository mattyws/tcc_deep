import gensim
import nltk
from DeepLearning import database, learn
from DeepLearning.helper import dictionary_to_list
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

word2vec_trainer = learn.Word2VecTrainer()
word2vec_model = word2vec_trainer.load_model('word2vec.model')
word2vec_model2 = word2vec_trainer.load_model('word2vec_mongo.model')
output_file = open('../word2vec_models/compare', 'w ')

# question = open('../TrainedLSTM/question-words.txt')

#
# some_vector = word2vec_model.wv['big'] - word2vec_model.wv['biggest'] + word2vec_model.wv['small']
# print(word2vec_model.wv.most_similar(positive=['big', 'small'], negative=['biggest']))

print("===================================== First Model ==========================================")
print("Vocabulary length: {}".format(len(word2vec_model.wv.vocab)))
output_file.write("Vocabulary length: " + str(len(word2vec_model.wv.vocab) ))
accuracy = word2vec_model.wv.accuracy('../TrainedLSTM/question-words.txt')

sum_corr = len(accuracy[-1]['correct'])
sum_incorr = len(accuracy[-1]['incorrect'])
total = sum_corr + sum_incorr
percent = lambda a: a / total * 100


print('Total sentences: {}\nCorrect: {:.2f}%\n Incorrect: {:.2f}%\n'.format(total, percent(sum_corr), percent(sum_incorr)))
output_file.write('Total sentences: ' + str(total) + '\nCorrect: ' +  str(percent(sum_corr)) +'\nIncorrect:' + str(percent(sum_incorr) + '\n'))

vocab1 = set(word2vec_model.wv.vocab.keys())

print("===================================== Second Model ==========================================")
print("Vocabulary length: {}".format(len(word2vec_model2.wv.vocab)))
output_file.write("Vocabulary length: " + str(len(word2vec_model.wv.vocab) ))
accuracy = word2vec_model2.wv.accuracy('../TrainedLSTM/question-words.txt')

sum_corr = len(accuracy[-1]['correct'])
sum_incorr = len(accuracy[-1]['incorrect'])
total = sum_corr + sum_incorr
percent = lambda a: a / total * 100


print('Total sentences: {}\nCorrect: {:.2f}%\n Incorrect: {:.2f}%\n'.format(total, percent(sum_corr), percent(sum_incorr)))
output_file.write('Total sentences: ' + str(total) + '\nCorrect: ' +  str(percent(sum_corr)) +'\nIncorrect:' + str(percent(sum_incorr) + '\n'))

vocab2 = set(word2vec_model2.wv.vocab.keys())

dif = vocab1.difference(vocab2)

print("Total words that exists in the first model but not in the second one: {}\n".format(len(dif)))
output_file.write("Total words that exists in the first model but not in the second one: "+str(len(dif)) + '\n')

print(dif)
output_file.write(dif)
