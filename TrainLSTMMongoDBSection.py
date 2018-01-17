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
training_documents_collection = 'training_embedding_old'
testing_documents_collection = 'testing_embedding_old'
model_saved_name = "../TrainedCNN/keras_nn_old.model"
result_file_name = "../TrainedCNN/result_nn_old"
epochs = 2
layers = 2


mongodb = MongoLoadDocumentMeta('patents')
documents = mongodb.get_all_meta(training_documents_collection)


result_string = ""

print("=============================== Filtering data and performing operations ===============================")
# Gets the first letter for the first IPC class and adding it to the ipc_sectons set variable
ipc_sections = set()
for doc in documents:
    if len(doc['ipc_classes']) > 0:
        ipc_sections.add(doc['ipc_classes'][0][0])
    else:
        print(doc['filename'])

# Creating a class_map variable, which contains a mapping of the IPC class to a number. The classes are ordered inside
# the classMap method. This mapping is important because of keras library particularities.
class_map = classMap(list(ipc_sections))


# Rebooting mongodb cursor
training_documents = mongodb.get_all_meta(training_documents_collection)

# The Generator for metadata and word embedding, its a python generator that returns "embeding, ipc_class
embedding_generator = MongoDBMetaEmbeddingGenerator(documents, "section", class_map, len(ipc_sections),
                                                    serve_forever=True, reshape=True)
print("=============================== Create training classes ===============================")
#Build a factory for a model adapter
model_factory = dl.factory.factory.create('KerasCovolutionalNetwork', input_shape=(maxWords, embeddingSize))
# model_factory = dl.factory.factory.create('MultilayerKerasRecurrentNN', input_shape=(maxWords, embeddingSize),
#                                                   numNeurouns=len(ipc_sections), numOutputNeurons=len(ipc_sections), layers=layers, use_dropout=True, dropout=0.5)
# model_factory = dl.factory.factory.create('KerasMultilayerPerceptron', num_class=len(ipc_sections), input_dim=200, layers=1,
#                                           hidden_units=[20], use_dropout=True, dropout=0.5)
model = model_factory.create()

timer.start() #start a timer for training
print("=============================== Training model, may take a while ===============================")
model.fit_generator(embedding_generator, batch_size=training_documents.count(), epochs=epochs) # start a training using the generator
timer.end() # ending the timer
result_string += "Total time to fit data : " + timer.elapsed() + "\n" # a information string to put in a file
print("Total time to fit data: " + timer.elapsed() + "\n")

print("=============================== Saving Model ===============================")
model.save(model_saved_name) # saving the model

# model = model.load(model_saved_name)

# Geting the test documents collection
test_documents = mongodb.get_all_meta(testing_documents_collection)
test_embedding_generator = MongoDBMetaEmbeddingGenerator(test_documents, "section", class_map, len(ipc_sections))


print("=============================== Predicting test data ===============================")
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

