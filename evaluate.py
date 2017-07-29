import os
from random import shuffle

import gensim
import nltk
from sklearn.metrics.classification import accuracy_score, recall_score, precision_score, f1_score

import DeepLearning as dl
import numpy as np

from DeepLearning.helper import *

timer = TimerCounter()
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
stop_set = nltk.corpus.stopwords.words('english')
stemmer = gensim.parsing.PorterStemmer()
maxWords = 150
embeddingSize = 200
x_data_path = "/tmp/x_word_embedding_"
y_data_path = "/tmp/y_labels_"

# Getting the hierarchcal structures from the database, and looping over it
train_data = dl.database.FlatStructureDatabase('../database/descriptions/base100')
test_data = dl.database.FlatStructureDatabase('../database/descriptions/testFiles')
keys = None
result_string = ""
for level, test_level in zip(train_data, test_data):
    class_map = classMap(list(level.keys()))
    #If keys is not None, so we already train for the section level
    if keys:
        data_paths = separate_class(keys, level)
        test_data_paths = separate_class(keys, test_level)
    else:
        data_paths = [level]
        test_data_paths = [test_level]
    #Looping through the files
    for data_dict, test_data_dict in zip(data_paths, test_data_paths):
        num_classes = len(list(data_dict.keys()))
        pathname = get_pathname(keys, data_dict)
        result_string += pathname + "\n\n"
        word2vec_model = dl.learn.Word2VecTrainer().load_model('word2vec.model')
        # doc2vec_model = dl.learn.Doc2VecTrainer().load_model('doc2vec.model')
        x_transformer = dl.data_representation.Word2VecEmbeddingCreator(word2vec_model, maxWords=maxWords)
        y_transformer = dl.data_representation.LabelsCreator(class_map, num_classes=num_classes, labels_to_categorical=True)
        dataP = dictionary_to_list(data_dict)
        shuffle(dataP)
        print(len(dataP))
        test_dataP = dictionary_to_list(test_data_dict)
        shuffle(test_dataP)
        print(len(dataP))

        print("=============================== Loading Word2Vec and Doc2Vec models ===============================")

        x = dl.database.XGenerator(x_transformer, dl.database.LoadTextCorpus(dataP, tokenizer=tokenizer, stop_set=stop_set), loop_forever=False)

        y = dl.database.YGenerator(y_transformer, dl.database.LoadTextCorpus(dataP, tokenizer=tokenizer, stop_set=stop_set), loop_forever=False)

        len_data = len(x)
        test_x = dl.database.XGenerator(x_transformer, dl.database.LoadTextCorpus(test_dataP, tokenizer=tokenizer, stop_set=stop_set), loop_forever=True)

        test_y = dl.database.YGenerator(y_transformer, dl.database.LoadTextCorpus(test_dataP, tokenizer=tokenizer, stop_set=stop_set), loop_forever=True)

        timer.start()
        if not os.path.exists(y_data_path+pathname):
            x_data_saver = dl.database.ObjectDatabaseSaver(x_data_path+pathname)
            y_data_saver = dl.database.ObjectDatabaseSaver(y_data_path+pathname)
            print("=============================== Dumping data representation on file ===============================")
            for data, label in zip(x, y):
                x_data_saver.save(data)
                y_data_saver.save(label)
        timer.end()
        result_string += "Total time to dump data : " + timer.elapsed() + "\n"

        timer.start()
        x_data_loader = dl.database.ObjectDatabaseReader(x_data_path+pathname, serve_forever=True)
        y_data_loader = dl.database.ObjectDatabaseReader(y_data_path+pathname, serve_forever=True)
        # for x in y_data_loader:
        #     print(x)
        model_factory = dl.factory.factory.create('MultilayerKerasRecurrentNN', input_shape=(maxWords, embeddingSize),
                                                  numNeurouns=num_classes, numOutputNeurons=num_classes, layers=2)

        model = model_factory.create()
        print("=============================== Training model ===============================")
        model.fit(x_data_loader, y_data_loader, batch_size=len(x), epochs=10)
        timer.end()
        result_string += "Total time to fit data : " + timer.elapsed() + "\n"

        print("=============================== Saving Model ===============================")
        model.save("kera_rnn_"+pathname+".model")

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
        print("Accuracy " + str(accuracy), "Recall " + str(recall), "Precision " + str(precision), "F1 " + str(f1))
        # f = open("result", "w")
        result_string += "Accuracy " + str(accuracy) + " Recall " + str(recall) + " Precision " + str(precision) + " F1 " + str(f1) + "\n"
        # f.close()


        # print("=============================== Initiating cross-validation ===============================")
        # k_fold_evaluate = dl.learn.KFoldEvaluate()
        # k_fold_evaluate.initiate(model_factory, x, y, numClasses=num_classes, epochs=10)
        # print("Results from the cross-validation")
        # k_fold_evaluate.print_mean()

    keys = level.keys()
    break
f = open("result", "w")
f.write(result_string)
f.close()