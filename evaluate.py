import os
from random import shuffle

import gensim
import nltk
from sklearn.metrics.classification import accuracy_score, recall_score, precision_score, f1_score

import DeepLearning as dl
import numpy as np

from DeepLearning.helper import *

tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
stop_set = nltk.corpus.stopwords.words('english')
stemmer = gensim.parsing.PorterStemmer()
maxWords = 150
embeddingSize = 200

# Getting the hierarchcal structures from the database, and looping over it
train_data = dl.database.FlatStructureDatabase('../database/descriptions/descriptions50')
test_data = dl.database.FlatStructureDatabase('../database/descriptions/testFiles3')
keys = None
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
        word2vec_model = dl.learn.Word2VecTrainer().load_model('word2vec.model')
        doc2vec_model = dl.learn.Doc2VecTrainer().load_model('doc2vec.model')
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

        if not os.path.exists("/tmp/y_labels"):
            x_data_saver = dl.database.ObjectDatabaseSaver("/tmp/x_word_embedding")
            y_data_saver = dl.database.ObjectDatabaseSaver("/tmp/y_labels")
            print("=============================== Dumping data representation on file ===============================")
            for data, label in zip(x, y):
                x_data_saver.save(data)
                y_data_saver.save(label)
        x_data_loader = dl.database.ObjectDatabaseReader("/tmp/x_word_embedding", serve_forever=True)
        y_data_loader = dl.database.ObjectDatabaseReader("/tmp/y_labels", serve_forever=True)
        # for x in y_data_loader:
        #     print(x)
        model_factory = dl.factory.factory.create('SimpleKerasRecurrentNN', input_shape=(maxWords, embeddingSize),
                                                  numNeurouns=embeddingSize, numOutputNeurons=num_classes)

        # model = model_factory.create()
        # print("=============================== Training model ===============================")
        # model.fit(x_data_loader, y_data_loader, batch_size=len(x), epochs=5)
        # print("=============================== Predicting test data ===============================")
        # pred = model.predict(test_x,test_y, batch_size=len(test_x))
        # real = []
        # print("Before")
        # i = 0
        # for i in range(0, len(test_y)):
        #     d = next(test_y)
        #     real.append(np.argmax(d))
        # accuracy = accuracy_score(real, pred)
        # recall = recall_score(real, pred, average='weighted')
        # precision = precision_score(real, pred, average='weighted')
        # f1 = f1_score(real, pred, average='weighted')
        # print("Accuracy " + str(accuracy), "Recall " + str(recall), "Precision " + str(precision), "F1 " + str(f1))
        # f = open("result", "w")
        # f.write("Accuracy " + str(accuracy) + " Recall " + str(recall) + " Precision " + str(precision) + " F1 " + str(f1))
        # f.close()


        # print("=============================== Initiating cross-validation ===============================")
        # k_fold_evaluate = dl.learn.KFoldEvaluate()
        # k_fold_evaluate.initiate(model_factory, x, y, numClasses=num_classes, epochs=10)
        # print("Results from the cross-validation")
        # k_fold_evaluate.print_mean()

    keys = level.keys()
    break