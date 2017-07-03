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


train_model = True
try:
  opts, args = getopt.getopt(sys.argv[1:], "hm:")
except getopt.GetoptError:
  print ('test.py -m <model_file>')
  sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print ('test.py -m <model_file>')
        sys.exit()
    elif opt in ("-m", "--model"):
        inputfile = arg
        print(inputfile)
        model = load_model(inputfile)


tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
stop_set = nltk.corpus.stopwords.words('english')
stemmer = gensim.parsing.PorterStemmer()
maxWords = 20
embeddingSize = 200

timer = TimerCounter()
# Getting the hierarchcal structures from the database, and looping over it
data_dict = dl.database.FlatStructureDatabase('../database/descriptions/base100').subclasses()
test_data_dict = dl.database.FlatStructureDatabase('../database/descriptions/testFiles3').subclasses()
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
print(os.path.exists("/tmp/y_labels50"))
if not os.path.exists("/tmp/y_labels50"):
    x_data_saver = dl.database.ObjectDatabaseSaver("/tmp/x_word_embedding50")
    y_data_saver = dl.database.ObjectDatabaseSaver("/tmp/y_labels50")
    print("=============================== Dumping data representation on file ===============================")
    for data, label in zip(x, y):
        x_data_saver.save(data)
        y_data_saver.save(label)
x_data_loader = dl.database.ObjectDatabaseReader("/tmp/x_word_embedding50", serve_forever=True)
y_data_loader = dl.database.ObjectDatabaseReader("/tmp/y_labels50", serve_forever=True)
timer.end()
result_string = "Total time to dump data : " + timer.elapsed() + "\n"

timer.start()
model_factory = dl.factory.factory.create('MultilayerKerasRecurrentNN', input_shape=(maxWords, embeddingSize),
                                          numNeurouns=50, numOutputNeurons=num_classes)

model = model_factory.create()
model.fit(x_data_loader, y_data_loader, batch_size=len(x), epochs=10)
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