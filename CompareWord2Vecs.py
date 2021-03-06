import gensim
import nltk
from DeepLearning import database, learn
from DeepLearning.database import MongoLoadDocumentMeta, MongoLoadDocumentData
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

word2vec_trainer = learn.Word2VecTrainer()
word2vec_models = ['../word2vec_models/word2vec_mongo_300.model']# '../word2vec_models/word2vec_old_300.model', '../word2vec_models/word2vec.model', '../word2vec_models/word2vec_mongo.model']#, '../word2vec_models/word2vec_50.model',
#                    '../word2vec_models/word2vec_50_mongo.model', '../word2vec_models/word2vec_400.model',
#                    '../word2vec_models/word2vec_400_mongo.model']
# word2vec_models = ['GoogleNews-vectors-negative300.bin']
mongodb = MongoLoadDocumentMeta('patents')
documents = mongodb.get_all_meta('training_docs100')
corpus = MongoLoadDocumentData('patents', documents, clean_text=True, tokenizer=tokenizer, stop_set=stop_set,description=True)

i=0
database_vocabulary = set()
for document in documents:
    # print(document['filename'])
    if i%1000 == 0:
        print(str(i) + ' ' + document['filename'])
    content = corpus.get_file_content(document['filename'])
    content = corpus.clean(content['description'])
    for token in content:
        database_vocabulary.add(token)
    i+=1
print("Training Database vocabulary size: " + str(len(database_vocabulary)))
for word2vec_model in word2vec_models:
    print("===================================== " + word2vec_model + " ==========================================")
    if 'Google' in word2vec_model:
        word2vec_model = word2vec_trainer.load_google_model(word2vec_model)
    else:
        word2vec_model = word2vec_trainer.load_model(word2vec_model)
    vocabulary = set(word2vec_model.wv.vocab.keys())
    print("Vocabulary size: " + str(len(vocabulary)))
    print("Words from database that dont exists in vocabulary: " + str(len(database_vocabulary.difference(vocabulary))))
    print("Words from vocabulary that dont exists in database: " + str(len(vocabulary.difference(database_vocabulary))))
    accuracy = word2vec_model.wv.accuracy('../TrainedLSTM/question-words.txt')
    sum_corr = len(accuracy[-1]['correct'])
    sum_incorr = len(accuracy[-1]['incorrect'])
    total = sum_corr + sum_incorr
    percent = lambda a: a / total * 100
    print('Total sentences: {}\nCorrect: {:.2f}%\n Incorrect: {:.2f}%\n'.format(total, percent(sum_corr), percent(sum_incorr)))

# print("===================================== Second Model ==========================================")
# print("Vocabulary length: {}".format(len(word2vec_model2.wv.vocab)))
# output_file.write("Vocabulary length: " + str(len(word2vec_model2.wv.vocab)) + '\n' )
# accuracy = word2vec_model2.wv.accuracy('../TrainedLSTM/question-words.txt')
#
# sum_corr = len(accuracy[-1]['correct'])
# sum_incorr = len(accuracy[-1]['incorrect'])
# total = sum_corr + sum_incorr
# percent = lambda a: a / total * 100
#
#
# print('Total sentences: {}\nCorrect: {:.2f}%\n Incorrect: {:.2f}%\n'.format(total, percent(sum_corr), percent(sum_incorr)))
# output_file.write('Total sentences: ' + str(total) + '\nCorrect: ' +  str(percent(sum_corr)) +'\nIncorrect:' + str(percent(sum_incorr)) + '\n')
#
# vocab2 = set(word2vec_model2.wv.vocab.keys())
#
# dif = vocab1.difference(vocab2)
# dif2 = vocab2.difference(vocab1)
#
# print("Total words that exists in the first model but not in the second one: {}\n".format(len(dif)))
# print("Total words that exists in the second model but not in the first one: "+str(len(dif2)) + '\n')
# output_file.write("Total words that exists in the first model but not in the second one: "+str(len(dif)) + '\n')
# output_file.write("Total words that exists in the second model but not in the first one: "+str(len(dif2)) + '\n')
#
# dif3 = database_vocabulary.difference(vocab1)
# dif4 = database_vocabulary.difference(vocab2)
# print("Total database vocabulary: " + str(len(database_vocabulary)) + '\n')
# output_file.write("Total database vocabulary: " + str(len(database_vocabulary)) + '\n')
# print("Total words that exists in the database but no in the first model: {}\n".format(len(dif3)))
# print("Total words that exists in the database but no in the second model: {}\n".format(len(dif4)))
# output_file.write("Total words that exists in the database but no in the first model: "+str(len(dif3)) + '\n')
# output_file.write("Total words that exists in the database but no in the second model: "+str(len(dif4)) + '\n')
