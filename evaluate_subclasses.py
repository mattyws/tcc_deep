import os
from random import shuffle

import gensim
import nltk
from sklearn.metrics.classification import accuracy_score, recall_score, precision_score, f1_score

import DeepLearning as dl
import numpy as np

from DeepLearning.helper import *
from keras.models import load_model
import sys
import getopt

def create_tmp_files(x_data_file, y_labels_file, x, y):
    x_data_saver = dl.database.ObjectDatabaseSaver(x_data_file)
    y_data_saver = dl.database.ObjectDatabaseSaver(y_labels_file)
    for data, label in zip(x, y):
        x_data_saver.save(data)
        y_data_saver.save(label)


def train_model(model, x, y, epochs=10):
    model.fit(x, y, batch_size=len(x), epochs=epochs)


try:
  opts, args = getopt.getopt(sys.argv[1:], "htm:")
except getopt.GetoptError:
  print ('test.py -m <model_file> -t')
  sys.exit(2)


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
    elif opt in ("-t", "--retrain"):
        retrain = True

y_labels_file = '/tmp/y_labels100'
x_data_file = '/tmp/x_word_embedding100'
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
stop_set = nltk.corpus.stopwords.words('english')
stemmer = gensim.parsing.PorterStemmer()
maxWords = 150
embeddingSize = 200

timer = TimerCounter()
# Getting the hierarchcal structures from the database, and looping over it
data_dict = dl.database.FlatStructureDatabase('../database/descriptions/descriptions50').subclasses()
test_data_dict = dl.database.FlatStructureDatabase('../database/descriptions/testFiles').subclasses()
keys = None


class_map = classMap(list(data_dict.keys()))

num_classes = len(list(data_dict.keys()))
pathname = get_pathname(keys, data_dict)
word2vec_model = dl.learn.Word2VecTrainer().load_model('word2vec.model')
doc2vec_model = dl.learn.Doc2VecTrainer().load_model('doc2vec.model')
x_transformer = dl.data_representation.Word2VecEmbeddingCreator(word2vec_model, maxWords=maxWords)
y_transformer = dl.data_representation.LabelsCreator(class_map, num_classes=num_classes, labels_to_categorical=True)
dataP = dictionary_to_list(data_dict)
shuffle(dataP)
test_dataP = dictionary_to_list(test_data_dict)
shuffle(test_dataP)

print("=============================== Loading Word2Vec and Doc2Vec models ===============================")

x = dl.database.XGenerator(x_transformer, dl.database.LoadTextCorpus(dataP, tokenizer=tokenizer, stop_set=stop_set), loop_forever=False)

y = dl.database.YGenerator(y_transformer, dl.database.LoadTextCorpus(dataP, tokenizer=tokenizer, stop_set=stop_set), loop_forever=False)

len_data = len(x)
test_x = dl.database.XGenerator(x_transformer, dl.database.LoadTextCorpus(test_dataP, tokenizer=tokenizer, stop_set=stop_set), loop_forever=True)

test_y = dl.database.YGenerator(y_transformer, dl.database.LoadTextCorpus(test_dataP, tokenizer=tokenizer, stop_set=stop_set), loop_forever=True)

timer.start()
if not os.path.exists(y_labels_file):
    print("=============================== Dumping data representation on file ===============================")
    create_tmp_files(x_data_file, y_labels_file, x, y)
x_data_loader = dl.database.ObjectDatabaseReader(x_data_file, serve_forever=True)
y_data_loader = dl.database.ObjectDatabaseReader(y_labels_file, serve_forever=True)
timer.end()
result_string = "Total time to dump data : " + timer.elapsed() + "\n"

if new_model:
    timer.start()
    model_factory = dl.factory.factory.create('SimpleKerasRecurrentNN', input_shape=(maxWords, embeddingSize),
                                              numNeurouns=num_classes, numOutputNeurons=num_classes, use_dropout=True)
    model = model_factory.create()
    print("=============================== Training Model ===============================")
    train_model(model, x, y)
    timer.end()
    result_string += "Total time to fit data : " + timer.elapsed() + "\n"
else:
    model = load_model(input_model_file)
    if retrain:
        timer.start()
        train_model(model, x, y)
        timer.end()
        result_string += "Total time to fit data : " + timer.elapsed() + "\n"

print("=============================== Saving Model ===============================")
model.save("kera_rnn.model")

print("=============================== Predicting test data ===============================")
pred = model.predict(test_x, batch_size=len(test_x))
real = []
print("Before")
i = 0
for i in range(0, len(test_y)):
    d = next(test_y)
    real.append(np.argmax(d))
accuracy = accuracy_score(real, pred)
recall = recall_score(real, pred, average='weighted')
precision = precision_score(real, pred, average='weighted')
f1 = f1_score(real, pred, average='weighted')
print(result_string + "Accuracy " + str(accuracy), "Recall " + str(recall), "Precision " + str(precision), "F1 " + str(f1))
f = open("result", "w")
f.write(result_string+"Accuracy " + str(accuracy) + " Recall " + str(recall) + " Precision " + str(precision) + " F1 " + str(f1))
f.close()
