import os
import pickle
from random import shuffle

import bson
import gensim
import nltk
import pandas as pd
import pymongo
from sklearn.metrics.classification import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

import DeepLearning as dl
from DeepLearning.database import MongoLoadDocumentData
from DeepLearning.database import MongoLoadDocumentMeta, MongoDBMetaEmbeddingGenerator
from DeepLearning.helper import classMap

# For LSTM models
# embedding_models= ['../word2vec_models/word2vec.model', '../word2vec_models/word2vec_50.model',
#                    '../word2vec_models/word2vec_50_mongo.model', '../word2vec_models/word2vec_400.model',
#                    '../word2vec_models/word2vec_400_mongo.model', '../word2vec_models/word2vec_mongo.model']
# embedding_sizes = [200, 50,
#                    50, 400,
#                    400, 200]
# classification_models = ['../TrainedLSTM/keras_rnn_shuffled_old.model', '../TrainedLSTM/keras_rnn_old_50.model',
#                          '../TrainedLSTM/keras_rnn_mongo_50.model', '../TrainedLSTM/keras_rnn_old_400.model',
#                          '../TrainedLSTM/keras_rnn_mongo_400.model', '../TrainedLSTM/keras_rnn_shuffled_mongo.model']
# test_databases=['testing_embedding_old', 'testing_embedding_old_50', 'testing_embedding_mongo_50',
#                 'testing_embedding_old_400', 'testing_embedding_mongo_400', 'testing_embedding_mongo']
# training_accuracies_overtime = [
#     [0.3379, 0.3939, 0.4105, 0.4222, 0.4398, 0.4620, 0.4766, 0.4856, 0.4945, 0.5007, 0.5097, 0.5177,
#      0.5269, 0.5274, 0.5321, 0.5390, 0.5403, 0.5436, 0.5465, 0.5484],
#     [0.3087, 0.3708, 0.3860, 0.4150, 0.4351, 0.4490, 0.4544, 0.4575, 0.4614, 0.4667, 0.4673, 0.4700,
#      0.4718, 0.4710, 0.4767, 0.4793, 0.4790, 0.4801, 0.4801, 0.4839],
#     [0.3091, 0.3539, 0.3894, 0.4166, 0.4318, 0.4412, 0.4476, 0.4548, 0.4579, 0.4646, 0.4680, 0.4703,
#      0.4722, 0.4753, 0.4787, 0.4790, 0.4826, 0.4853, 0.4892, 0.4928],
#     [0.2608, 0.2679, 0.2679, 0.2724, 0.2856, 0.2908, 0.2915, 0.2950, 0.2986, 0.3016, 0.3051, 0.3079,
#      0.3116, 0.3140, 0.3153, 0.3175, 0.3184, 0.3211, 0.3211, 0.3258],
#     [0.2697, 0.2795, 0.2927, 0.2970, 0.2996, 0.3014, 0.3036, 0.3049, 0.3062, 0.3072, 0.3074, 0.3087,
#      0.3090, 0.3095, 0.3136, 0.3163, 0.3192, 0.3215, 0.3230, 0.3237],
#     [0.2662, 0.2671, 0.2671, 0.2671, 0.2671, 0.2671, 0.2671, 0.2671, 0.2671, 0.2671, 0.2671, 0.2671,
#      0.2671, 0.2671, 0.2671, 0.2671, 0.2671, 0.2671, 0.2671, 0.2671],
# ]

# For multilayer perceptron
embedding_models = [
    # '../doc2vec_models/doc2vec_mongo.model', '../doc2vec_models/doc2vec_mongo_50.model',
    # '../doc2vec_models/doc2vec_mongo_300.model',
    # '../doc2vec_models/doc2vec_mongo_400.model',
    '../doc2vec_models/doc2vec_old.model','../doc2vec_models/doc2vec_old_50.model',
    '../doc2vec_models/doc2vec_old_300.model', '../doc2vec_models/doc2vec_old_400.model'
]
embedding_sizes = [
    # 200, 50, 300,
    # 400,
    200, 50, 300, 400
]
classification_models = [
    # '..//TrainedNN/keras_nn_mongo.model', '../TrainedNN/keras_nn_mongo_50.model',
    # '../TrainedNN/keras_nn_mongo_300.model',
    # '../TrainedNN/keras_nn_mongo_400.model',
    '../TrainedNN/keras_nn_old.model',
    '../TrainedNN/keras_nn_old_50.model', '../TrainedNN/keras_nn_old_300.model', '../TrainedNN/keras_nn_old_400.model'
]
test_databases = [
    # 'testing_embedding_mongo_200', 'testing_embedding_mongo_50', 'testing_embedding_mongo_300',
    # 'testing_embedding_mongo_400',
    'testing_embedding_old_200', 'testing_embedding_old_50',
    'testing_embedding_old_300', 'testing_embedding_old_400'
]
training_accuracies_overtime = [
    # [0.2634, 0.2708, 0.2715, 0.2719, 0.2758, 0.2784, 0.2795, 0.2814, 0.2814, 0.2813, 0.2836, 0.2812],
    # [0.2788, 0.2789, 0.2800, 0.2826, 0.2831, 0.2849, 0.2860, 0.2869, 0.2883, 0.2911, 0.2918, 0.2929],
    # [0.2777, 0.2714, 0.2749, 0.2791, 0.2769, 0.2775, 0.2836, 0.2858, 0.2887, 0.2877, 0.2892, 0.2889],
    # [0.2761, 0.2768, 0.2781, 0.2771, 0.2796, 0.2847, 0.2856, 0.2858, 0.2877, 0.2888, 0.2905, 0.2916],
    [0.2739, 0.2739, 0.2759, 0.2793, 0.2804, 0.2835, 0.2824, 0.2859, 0.2864, 0.2868, 0.2889, 0.2901],
    [0.2786, 0.2758, 0.2789, 0.2815, 0.2830, 0.2850, 0.2865, 0.2876, 0.2871, 0.2892, 0.2897, 0.2920],
    [0.2774, 0.2765, 0.2776, 0.2790, 0.2824, 0.2846, 0.2851, 0.2878, 0.2877, 0.2883, 0.2895, 0.2914],
    [0.2763, 0.2783, 0.2785, 0.2826, 0.2797, 0.2810, 0.2843, 0.2849, 0.2855, 0.2889, 0.2877, 0.2892]
]


training_collection = 'testing_docs100'

'''
Configurations
'''
language = 'english'
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
stop_set = nltk.corpus.stopwords.words(language)
stemmer = gensim.parsing.PorterStemmer()
mongodb = MongoLoadDocumentMeta('patents')
max_words = 150
result_file_name = "result_model"

for embedding_model, classification_model, test_database, embedding_size, training_acc_overtime in \
        zip(embedding_models, classification_models, test_databases, embedding_sizes, training_accuracies_overtime):
    documents = mongodb.get_all_meta(training_collection)
    corpus = MongoLoadDocumentData('patents', documents, clean_text=True, tokenizer=tokenizer, stop_set=stop_set,description=True)

    # word2vec_model = dl.learn.Word2VecTrainer().load_model(embedding_model)
    # word_vector_generator = dl.data_representation.Word2VecEmbeddingCreator(word2vec_model, maxWords=max_words, embeddingSize=embedding_size)

    doc2vec_model = dl.learn.Doc2VecTrainer().load_model(embedding_model)
    doc_vector_generator = dl.data_representation.Doc2VecEmbeddingCreator(doc2vec_model)
    print("=============================== Shuffling test database for " + classification_model.split('/')[-1] + " ===============================")
    shuffled = []
    for document in documents:
        shuffled.append(document["filename"])
    shuffle(shuffled)
    print("=============================== Creating test database for " + classification_model.split('/')[-1] + " ===============================")
    i=0
    client = pymongo.MongoClient()
    patents_database = client.patents
    word_embedding_collection = patents_database[test_database]
    for doc in shuffled:
        document = mongodb.get_document_by(training_collection, 'filename', doc)
        # print(document['filename'])
        if i%1000 == 0:
            print(str(i) + ' ' + document['filename'])
        content = corpus.get_file_content(document['filename'])
        content = corpus.clean(content['description'])
        # word_embedding_matrix = word_vector_generator.create_x_text(content)
        word_embedding_matrix = doc_vector_generator.create_x_text(content).reshape((1,embedding_size))
        document['embedding'] = bson.binary.Binary(pickle.dumps(word_embedding_matrix, protocol=2))
        word_embedding_collection.insert_one(document)
        i+=1
        # fs = gridfs.GridFS(patents_database, collection="documents_embedding_docs100")
        # with fs.new_file(filename=document['filename'], content_type="binary") as fp:
        #     fp.write(word_embedding_matrix)
        # print(content)
    result_directory = "../TrainedNN/results/" + classification_model.split('/')[-1] +"/"


    if not os.path.exists(result_directory):
        os.mkdir(result_directory)

    documents = mongodb.get_all_meta('training_docs100')
    result_string = ""

    print("=============================== Filtering data and performing operations ===============================")
    # Gets the first letter for the first IPC class and adding it to the ipc_sectons set variable
    ipc_sections = set()
    for doc in documents:
        if len(doc['ipc_classes']) > 0:
            ipc_sections.add(doc['ipc_classes'][0][0])
        else:
            print(doc['filename'])
    print(ipc_sections)
    ipc_sections = list(ipc_sections)

    # Creating a class_map variable, which contains a mapping of the IPC class to a number. The classes are ordered inside
    # the classMap method. This mapping is important because of keras library particularities.
    class_map = classMap(ipc_sections)
    ipc_sections.sort()

    embedding_generator = MongoDBMetaEmbeddingGenerator(documents, "section", class_map, len(ipc_sections), serve_forever=True)
    print("=============================== Create training classes ===============================")
    #Build a factory for a model adapter
    # model_factory = dl.factory.factory.create('MultilayerKerasRecurrentNN', input_shape=(max_words, embedding_size),
    #                                                   numNeurouns=len(ipc_sections), numOutputNeurons=len(ipc_sections), layers=1)
    model_factory = dl.factory.factory.create('KerasMultilayerPerceptron', num_class=len(ipc_sections), input_dim=embedding_size, layers=1,
                                          hidden_units=[25])
    model = model_factory.create()

    model = model.load(classification_model)

    # Geting the test documents collection
    test_documents = mongodb.get_all_meta(test_database)


    print("=============================== Predicting test data ===============================")
    # Predicting the class for each word vector in the database
    real = []
    all_class = []
    pred = []
    # for doc, ipc in test_embedding_generator:
    #     result = model.predict_one(doc)
    #     pred.append(class_map[result]) #adding the result to the predicted vector
    #     real.append(class_map[numpy.argmax(ipc)]) #Adding the real value to de real class vector

    for doc in test_documents:
        result = model.predict_one(pickle.loads(doc['embedding']))
        pred.append(class_map[result]) #adding the result to the predicted vector
        real.append(doc['ipc_classes'][0][0])
        document_section = set()
        for c in doc['ipc_classes']:
            document_section.add(c[0])
        if class_map[result] in document_section:
            all_class.append(doc['ipc_classes'][0][0])
        else:
            all_class.append(class_map[result])

    #Calculating the metric F1, Precision, Accuracy and Recall
    accuracy = accuracy_score(real, pred)
    recall = recall_score(real, pred, average='weighted')
    recall_per_class = recall_score(real, pred, average=None)
    precision = precision_score(real, pred, average='weighted')
    precision_per_class = precision_score(real, pred, average=None)
    f1 = f1_score(real, pred, average='weighted')
    f1_per_class = f1_score(real, pred, average=None)
    results_per_class = dict()
    for i in range(0, len(recall_per_class)):
        if not class_map[i] in results_per_class.keys():
            results_per_class[class_map[i]] = []
        results_per_class[class_map[i]].append(recall_per_class[i])
        results_per_class[class_map[i]].append(precision_per_class[i])
        results_per_class[class_map[i]].append(f1_per_class[i])

    #Calculating the metric F1, Precision, Accuracy and Recall for all classes
    accuracy_all_class = accuracy_score(real, all_class)
    recall_all_class = recall_score(real, all_class, average='weighted')
    precision_all_class = precision_score(real, all_class, average='weighted')
    f1_all_class = f1_score(real, all_class, average='weighted')

    matrix = confusion_matrix(real, pred, labels=ipc_sections.sort())

    #ploting

    ts = pd.Series(training_acc_overtime, index=range(len(training_acc_overtime)))
    plot = ts.plot(x='Iteração', y='Acurácia', ylim=(0, 1.0))
    fig = plot.get_figure()
    fig.savefig(result_directory+"training_acc_overtime.png")
    plot.clear()
    fig.clear()

    df2 = pd.DataFrame([results_per_class[x] for x in ipc_sections], index=ipc_sections ,columns=['Recall', 'Precisão', 'F-Score'])
    plot2 = df2.plot.bar(ylim=(0, 1.0))
    plot2.legend(ncol=3)
    fig2 = plot2.get_figure()
    fig2.savefig(result_directory+"result_per_class.png")
    plot2.clear()
    fig2.clear()



    print("Accuracy " + str(accuracy), "Recall " + str(recall), "Precision " + str(precision), "F1 " + str(f1))
    print("Accuracy_all_class " + str(accuracy_all_class), "Recall_all_class " + str(recall_all_class), "Precision_all_class "
          + str(precision_all_class), "F1_all_class " + str(f1_all_class) )
    result_string += "Accuracy " + str(accuracy) + " Recall " + str(recall) + " Precision " + str(precision) + " F1 " + str(f1) + "\n"
    result_string += "Accuracy_all_class " + str(accuracy_all_class) + "Recall_all_class " + str(recall_all_class) + "Precision_all_class "  \
          + str(precision_all_class) + "F1_all_class " + str(f1_all_class) + "\n"
    f = open(result_directory+result_file_name, "w")
    f.write("Database: " + test_database +"\n")
    f.write("embedding matrix: " + str(max_words) + "x" + str(embedding_size)+"\n")
    f.write(result_string)
    f.write("Recall Precisão F-Score")
    for c in results_per_class:
        f.write(c)
    f.close()
    word_embedding_collection.drop()

