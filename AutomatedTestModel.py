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
# embedding_models= [
#     # '../word2vec_models/word2vec.model',
#     # '../word2vec_models/word2vec_50.model',
#     # '../word2vec_models/word2vec_50_mongo.model',
#     # '../word2vec_models/word2vec_400.model',
#     # '../word2vec_models/word2vec_400_mongo.model',
#     # '../word2vec_models/word2vec_mongo.model',
#     # '../word2vec_models/GoogleNews-vectors-negative300.bin',
#     # '../word2vec_models/word2vec_old_300.model',
#     # '../word2vec_models/word2vec_mongo_300.model'
# ]
# embedding_sizes = [
#     # 200,
#     # 50,
#     # 50,
#     # 400,
#     # 400,
#     # 200,
#     # 300,
#     # 300,
#     # 300
# ]
# classification_models = [
#     # '../TrainedLSTM/keras_rnn_shuffled_old.model',
#     # '../TrainedLSTM/keras_rnn_old_50.model',
#     #  '../TrainedLSTM/keras_rnn_mongo_50.model',
#     # '../TrainedLSTM/keras_rnn_old_400.model',
#     #  '../TrainedLSTM/keras_rnn_mongo_400.model',
#     # '../TrainedLSTM/keras_rnn_shuffled_mongo.model',
#     # '../TrainedLSTM/keras_rnn_google.model',
#     # '../TrainedLSTM/keras_rnn_old_300.model',
#     # '../TrainedLSTM/keras_rnn_mongo_300.model'
#
# ]
# test_databases=[
#     # 'testing_embedding_old',
#     # 'testing_embedding_old_50',
#     # 'testing_embedding_mongo_50',
#     # 'testing_embedding_old_400',
#     # 'testing_embedding_mongo_400',
#     # 'testing_embedding_mongo',
#     # 'testing_embedding_google',
#     # 'testing_embedding_old_300',
#     # 'testing_embedding_mongo_300'
# ]
# training_accuracies_overtime = [
#     # [0.3379, 0.3939, 0.4105, 0.4222, 0.4398, 0.4620, 0.4766, 0.4856, 0.4945, 0.5007, 0.5097, 0.5177,
#     #  0.5269, 0.5274, 0.5321, 0.5390, 0.5403, 0.5436, 0.5465, 0.5484],
#     # [0.3087, 0.3708, 0.3860, 0.4150, 0.4351, 0.4490, 0.4544, 0.4575, 0.4614, 0.4667, 0.4673, 0.4700,
#     #  0.4718, 0.4710, 0.4767, 0.4793, 0.4790, 0.4801, 0.4801, 0.4839],
#     # [0.3091, 0.3539, 0.3894, 0.4166, 0.4318, 0.4412, 0.4476, 0.4548, 0.4579, 0.4646, 0.4680, 0.4703,
#     #  0.4722, 0.4753, 0.4787, 0.4790, 0.4826, 0.4853, 0.4892, 0.4928],
#     # [0.2608, 0.2679, 0.2679, 0.2724, 0.2856, 0.2908, 0.2915, 0.2950, 0.2986, 0.3016, 0.3051, 0.3079,
#     #  0.3116, 0.3140, 0.3153, 0.3175, 0.3184, 0.3211, 0.3211, 0.3258],
#     # [0.2697, 0.2795, 0.2927, 0.2970, 0.2996, 0.3014, 0.3036, 0.3049, 0.3062, 0.3072, 0.3074, 0.3087,
#     #  0.3090, 0.3095, 0.3136, 0.3163, 0.3192, 0.3215, 0.3230, 0.3237],
#     # [0.2662, 0.2671, 0.2671, 0.2671, 0.2671, 0.2671, 0.2671, 0.2671, 0.2671, 0.2671, 0.2671, 0.2671,
#     #  0.2671, 0.2671, 0.2671, 0.2671, 0.2671, 0.2671, 0.2671, 0.2671],
#     # [0.2673, 0.2679, 0.2679, 0.2679, 0.2679, 0.2679, 0.2679, 0.2679, 0.2679, 0.2679, 0.2679, 0.2679,
#     #  0.2679, 0.2679, 0.2679, 0.2679, 0.2679, 0.2679, 0.2679, 0.2679],
#     # [0.2683, 0.2721, 0.2926, 0.2899, 0.2900, 0.2958, 0.2984, 0.3024, 0.3045, 0.3040, 0.3054, 0.3101,
#     #  0.3102, 0.3148, 0.3152, 0.3193, 0.3229, 0.3255, 0.3315, 0.3297],
#     # [0.2757, 0.2961, 0.2955, 0.2986, 0.3029, 0.3069, 0.3124, 0.3159, 0.3205, 0.3210, 0.3248, 0.3258,
#     #  0.3290, 0.3319, 0.3349, 0.3385, 0.3435, 0.3483, 0.3532, 0.3549]
# ]

# For Conv nets
# embedding_models= [
#     # '../word2vec_models/word2vec_50_mongo.model',
#     # '../word2vec_models/word2vec_mongo.model',
#     # '../word2vec_models/word2vec_mongo_300.model',
#     # '../word2vec_models/word2vec_400_mongo.model',
#     # '../word2vec_models/word2vec_50.model',
#     # '../word2vec_models/word2vec.model',
#     # '../word2vec_models/word2vec_old_300.model',
#     # '../word2vec_models/word2vec_400.model',
#     # '../word2vec_models/GoogleNews-vectors-negative300.bin'
# ]
# embedding_sizes = [
#     # 50,
#     # 200,
#     # 300,
#     # 400,
#     # 50,
#     # 200,
#     # 300,
#     # 400,
#     # 300
# ]
# classification_models = [
#     # '../TrainedConv/keras_cnn_mongo_50.model',
#     # '../TrainedConv/keras_cnn_mongo_200.model',
#     # '../TrainedConv/keras_cnn_mongo_300.model',
#     # '../TrainedConv/keras_cnn_mongo_400.model',
#     # '../TrainedConv/keras_cnn_old_50.model',
#     # '../TrainedConv/keras_cnn_old_200.model',
#     # '../TrainedConv/keras_cnn_old_300.model',
#     # '../TrainedConv/keras_cnn_old_400.model',
#     # '../TrainedConv/keras_cnn_google.model'
# ]
# test_databases=[
#     # 'testing_embedding_mongo_50',
#     # 'testing_embedding_mongo'
#     # 'testing_embedding_mongo_300'
#     # 'testing_embedding_mongo_400',
#     # 'testing_embedding_old_50',
#     # 'testing_embedding_old',
#     # 'testing_embedding_old_300',
#     # 'testing_embedding_old_400',
#     # 'testing_embedding_google'
# ]
# training_accuracies_overtime = [
#     # [0.4509, 0.5103, 0.5354, 0.5583, 0.5796, 0.6019, 0.6195, 0.6378, 0.6576, 0.6736, 0.6879, 0.7017, 0.7169, 0.7286,
#     #  0.7377, 0.7492, 0.7571, 0.7646, 0.7726, 0.7802],
#     # [0.2663, 0.2666, 0.2665, 0.2667, 0.2671, 0.2670, 0.2671, 0.2671, 0.2671, 0.2671, 0.2671, 0.2671, 0.2671, 0.2671,
#     #  0.2671, 0.2671, 0.2671, 0.2671, 0.2671, 0.2671],
#     # [0.2685, 0.2929, 0.3009, 0.3020, 0.3092, 0.3245, 0.3135, 0.3492, 0.3737, 0.3835, 0.4855, 0.4935, 0.5103, 0.5274,
#     #  0.5460, 0.5596, 0.5754, 0.5887, 0.6043, 0.6168],
#     # [0.2681, 0.2709, 0.2821, 0.3057, 0.3296, 0.3493, 0.3659, 0.3832, 0.3977, 0.4126, 0.4268, 0.4429, 0.4588, 0.4760,
#     #  0.4928, 0.5062, 0.5237, 0.5405, 0.5565, 0.5717],
#     # [0.4573, 0.5222, 0.5470, 0.5686, 0.5901, 0.6112, 0.6335, 0.6494, 0.6679, 0.6862, 0.7033, 0.7163, 0.7296, 0.7414,
#     #  0.7504, 0.7612, 0.7699, 0.7768, 0.7836, 0.7910],
#     # [0.5605, 0.6373, 0.6658, 0.6933, 0.7202, 0.7470, 0.7742, 0.7972, 0.8198, 0.8370, 0.8510, 0.8631, 0.8767, 0.8862,
#     #  0.8951, 0.9015, 0.9089, 0.9160, 0.9210, 0.9256],
#     # [0.2668, 0.2785, 0.3103, 0.3361, 0.3543, 0.3706, 0.3880, 0.4043, 0.4225, 0.4396, 0.4596, 0.4761, 0.4984, 0.5170,
#     #  0.5336, 0.5538, 0.5674, 0.5847, 0.6035, 0.6167],
#     # [0.2670, 0.2694, 0.2789, 0.2932, 0.3091, 0.3228, 0.3355, 0.3523, 0.3683, 0.3848, 0.4024, 0.4237, 0.4427, 0.4656,
#     #  0.4864, 0.5067, 0.5266, 0.5503, 0.5659, 0.5879],
#     # [0.2669, 0.2674, 0.2677, 0.2677, 0.2676, 0.2678, 0.2678, 0.2678, 0.2679, 0.2679, 0.2679, 0.2679, 0.2679, 0.2679,
#     #  0.2679, 0.2679, 0.2679, 0.2679, 0.2679, 0.2679]
# ]

# For multilayer perceptron
embedding_models = [
    # '../doc2vec_models/doc2vec_mongo_200.model',
    # '../doc2vec_models/doc2vec_mongo_50.model',
    # '../doc2vec_models/doc2vec_mongo_300.model',
    '../doc2vec_models/doc2vec_mongo_400.model',
    '../doc2vec_models/doc2vec_old_200.model',
    '../doc2vec_models/doc2vec_old_50.model',
    '../doc2vec_models/doc2vec_old_300.model',
    '../doc2vec_models/doc2vec_old_400.model'
]
embedding_sizes = [
    # 200,
    # 50,
    # 300,
    400,
    200,
    50,
    300,
    400
]
classification_models = [
    # '..//TrainedNN/keras_nn_mongo_200.model',
    # '../TrainedNN/keras_nn_mongo_50.model',
    # '../TrainedNN/keras_nn_mongo_300.model',
    '../TrainedNN/keras_nn_mongo_400.model',
    '../TrainedNN/keras_nn_old_200.model',
    '../TrainedNN/keras_nn_old_50.model',
    '../TrainedNN/keras_nn_old_300.model',
    '../TrainedNN/keras_nn_old_400.model'
]
test_databases = [
    # 'testing_embedding_mongo_200',
    # 'testing_embedding_mongo_50',
    # 'testing_embedding_mongo_300',
    'testing_embedding_mongo_400',
    'testing_embedding_old_200',
    'testing_embedding_old_50',
    'testing_embedding_old_300',
    'testing_embedding_old_400'
]
training_accuracies_overtime = [
    # [0.3458, 0.3389, 0.3455, 0.3516, 0.3519, 0.3571, 0.3616, 0.3717, 0.3794, 0.3884, 0.3971, 0.4017],
    # [0.3967, 0.3859, 0.3846, 0.3845, 0.3951, 0.3941, 0.3879, 0.3811, 0.3864, 0.3940, 0.4124, 0.4293],
    # [0.3296, 0.3296, 0.3311, 0.3395, 0.3423, 0.3506, 0.3586, 0.3570, 0.3667, 0.3742, 0.3768, 0.3879],
    [0.2968, 0.3160, 0.3032, 0.2910, 0.2810, 0.3049, 0.3036, 0.3063, 0.3075, 0.3022, 0.3039, 0.3119],
    [0.3524, 0.3572, 0.3712, 0.3842, 0.3868, 0.3914, 0.3966, 0.4037, 0.4087, 0.4215, 0.4262, 0.4325],
    [0.3763, 0.3643, 0.3710, 0.3713, 0.3739, 0.3675, 0.3841, 0.3865, 0.3942, 0.4004, 0.4054, 0.4168],
    [0.3331, 0.3327, 0.3389, 0.3494, 0.3570, 0.3713, 0.3810, 0.3875, 0.3955, 0.3989, 0.4006, 0.4095],
    [0.2996, 0.3004, 0.3030, 0.3155, 0.3085, 0.3159, 0.3160, 0.3180, 0.3152, 0.3151, 0.3183, 0.2933]
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

    # if 'Google' in embedding_model:
    #     word2vec_model = dl.learn.Word2VecTrainer().load_google_model(embedding_model)
    # else:
    #     word2vec_model = dl.learn.Word2VecTrainer().load_model(embedding_model)
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
    result_directory = "../TrainedConv/results/" + classification_model.split('/')[-1] +"/"


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
    result_string += "Accuracy_all_class " + str(accuracy_all_class) + " Recall_all_class " + str(recall_all_class) + " Precision_all_class "  \
          + str(precision_all_class) + " F1_all_class " + str(f1_all_class) + "\n"
    f = open(result_directory+result_file_name, "w")
    f.write("Database: " + test_database +"\n")
    f.write("embedding matrix: " + str(max_words) + "x" + str(embedding_size)+"\n")
    f.write(result_string)
    f.write("Recall Precisão F-Score\n")
    for c in results_per_class.keys():
        f.write(c + " : " + str(results_per_class[c])+'\n')
    f.close()
    word_embedding_collection.drop()

