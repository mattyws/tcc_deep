'''
Script used to test a model for the IPC Section level.
The script load a keras trained model.
'''
import pickle

import numpy
from sklearn.metrics.classification import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

import DeepLearning as dl
from DeepLearning.database import MongoLoadDocumentMeta , MongoDBMetaEmbeddingGenerator

from DeepLearning.helper import TimerCounter, classMap
import os
import pandas as pd
import numpy as np



'''
Configurations
'''
maxWords = 150
embeddingSize = 200
timer = TimerCounter() # Timer to count how long it takes to perform each process
training_documents_collection = 'shuffled_training_embedding_old'
testing_documents_collection = 'testing_embedding_old_mongo'
model_saved_name = "../TrainedLSTM/keras_rnn_shuffled_mongo.model"
result_directory = "../TrainedLSTM/results/keras_rnn_shuffled_mongo/"
result_file_name = "result_rnn_word2vec_shuffled"
epochs = 12
layers = 2
training_acc_overtime = [0.2662, 0.2671, 0.2671, 0.2671, 0.2671, 0.2671, 0.2671, 0.2671, 0.2671, 0.2671, 0.2671, 0.2671,
                         0.2671,0.2671, 0.2671, 0.2671, 0.2671, 0.2671, 0.2671, 0.2671, 0.2671]

if not os.path.exists(result_directory):
    os.mkdir(result_directory)


mongodb = MongoLoadDocumentMeta('patents')
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
model_factory = dl.factory.factory.create('MultilayerKerasRecurrentNN', input_shape=(maxWords, embeddingSize),
                                                  numNeurouns=len(ipc_sections), numOutputNeurons=len(ipc_sections), layers=layers)
model = model_factory.create()

model = model.load(model_saved_name)

# Geting the test documents collection
test_documents = mongodb.get_all_meta(testing_documents_collection)
test_embedding_generator = MongoDBMetaEmbeddingGenerator(test_documents, "section", class_map, len(ipc_sections))


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
    all_class.append(doc['ipc_classes'])

print(pred)
print(real)

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


matrix = confusion_matrix(real, pred, labels=ipc_sections.sort())

#ploting

ts = pd.Series(training_acc_overtime, index=range(len(training_acc_overtime)))
plot = ts.plot(x='Iteração', y='Acurácia')
fig = plot.get_figure()
fig.savefig(result_directory+"training_acc_overtime.png")
df2 = pd.DataFrame([results_per_class[x] for x in ipc_sections], index=ipc_sections ,columns=['Recall', 'Precisão', 'F-Score'])
plot = df2.plot.bar()
fig = plot.get_figure()
fig.savefig(result_directory+"result_per_class.png")



print("Accuracy " + str(accuracy), "Recall " + str(recall), "Precision " + str(precision), "F1 " + str(f1))
result_string += "Accuracy " + str(accuracy) + " Recall " + str(recall) + " Precision " + str(precision) + " F1 " + str(f1) + "\n"
f = open(result_directory+result_file_name, "w")
f.write("Database: " + training_documents_collection +"\n")
f.write("embedding matrix: " + str(maxWords) + "x" + str(embeddingSize)+"\n")
f.write("epochs: " + str(epochs)+"\n")
f.write("layers : " + str(layers)+"\n")
f.write(result_string)
f.close()
