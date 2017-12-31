'''
Script used to train a model for the IPC Section level.
It train a model using a collection of the word embedding data (here specified by the 'documents_collection' variable),
and tests on another collection (using the 'testing_documents_collection' variable). It would be recomendable to use
different collection with different documents each.
'''

import numpy
from sklearn.metrics.classification import accuracy_score, recall_score, precision_score, f1_score

import DeepLearning as dl
from DeepLearning.database import MongoLoadDocumentMeta , MongoDBMetaEmbeddingGenerator

from DeepLearning.helper import TimerCounter, classMap



'''
Configurations
'''
maxWords = 150
embeddingSize = 200
timer = TimerCounter() # Timer to count how long it takes to perform each process
training_documents_collection = 'training_embedding_float'
testing_documents_collection = 'testing_embedding_float'
epochs = 12
layers = 2


mongodb = MongoLoadDocumentMeta('patents')
documents = mongodb.get_all_meta(testing_documents_collection)

print("=============================== Filtering data and performing operations ===============================")
# Gets the first letter for the first IPC class and adding it to the ipc_sectons set variable
ipc_classes = dict()
ipc_sections = set()
for doc in documents:
    if len(doc['ipc_classes']) > 0:
        ipc_sections.add(doc['ipc_classes'][0][0])
        if doc['ipc_classes'][0][0] not in ipc_classes.keys():
            ipc_classes[doc['ipc_classes'][0][0]] = []
        if doc['ipc_classes'][0][0:3] not in ipc_classes[doc['ipc_classes'][0][0]]:
            ipc_classes[doc['ipc_classes'][0][0]].append(doc['ipc_classes'][0][0:3])
    else:
        print(doc['filename'])
print(ipc_sections)
print(ipc_classes)

# Creating a class_map variable, which contains a mapping of the IPC class to a number. The classes are ordered inside
# the classMap method. This mapping is important because of keras library particularities.

for key in ipc_classes.keys():
    model_saved_name = "../TrainedLSTM/keras_rnn_mongo_float_"+key+".model"
    result_file_name = "../TrainedLSTM/result_rnn_mongo_float_"+key
    result_string = ""

    classes = ipc_classes[key]
    classes.sort()
    class_map = classMap(classes)
    documents = mongodb.get_meta_by_section(training_documents_collection, key)
    # Rebooting mongodb cursor
    training_documents = mongodb.get_all_meta(training_documents_collection)
    # The Generator for metadata and word embedding, its a python generator that returns "embeding, ipc_class
    embedding_generator = MongoDBMetaEmbeddingGenerator(documents, "class", class_map, len(classes), serve_forever=True)
    print("=============================== Create training classes " + key + " ===============================")
    #Build a factory for a model adapter
    model_factory = dl.factory.factory.create('MultilayerKerasRecurrentNN', input_shape=(maxWords, embeddingSize),
                                                      numNeurouns=len(classes), numOutputNeurons=len(classes), layers=layers, use_dropout=True)
    model = model_factory.create()

    timer.start() #start a timer for training
    print("=============================== Training model for " + key + ", may take a while ===============================")
    model.fit_generator(embedding_generator, batch_size=documents.count(), epochs=epochs) # start a training using the generator
    timer.end() # ending the timer
    result_string += "Total time to fit data : " + timer.elapsed() + "\n" # a information string to put in a file
    print("Total time to fit data: " + timer.elapsed() + "\n")

    print("=============================== Saving Model ===============================")
    model.save(model_saved_name) # saving the model

    # model = model.load(model_saved_name)

    # Geting the test documents collection
    test_documents = mongodb.get_meta_by_section(testing_documents_collection, key)
    test_embedding_generator = MongoDBMetaEmbeddingGenerator(test_documents, "class", class_map, len(classes))


    print("=============================== Predicting test data for "+key+" ===============================")
    # Predicting the class for each word vector in the database
    real = []
    pred = []
    for doc, ipc in test_embedding_generator:
        result = model.predict_one(doc)
        pred.append(class_map[result]) #adding the result to the predicted vector
        real.append(class_map[numpy.argmax(ipc)]) #Adding the real value to de real class vector

    #Calculating the metric F1, Precision, Accuracy and Recall
    accuracy = accuracy_score(real, pred)
    recall = recall_score(real, pred, average='weighted')
    precision = precision_score(real, pred, average='weighted')
    f1 = f1_score(real, pred, average='weighted')
    print("Accuracy " + str(accuracy), "Recall " + str(recall), "Precision " + str(precision), "F1 " + str(f1))
    result_string += "Accuracy " + str(accuracy) + " Recall " + str(recall) + " Precision " + str(precision) + " F1 " + str(f1) + "\n"
    f = open(result_file_name, "w")
    f.write("Database: " + training_documents_collection)
    f.write("embedding matrix: " + str(maxWords) + " " + str(embeddingSize))
    f.write("epochs: " + str(epochs))
    f.write("layers : " + str(layers))
    f.write(result_string)
    f.close()
