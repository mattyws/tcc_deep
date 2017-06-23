import abc

from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.neural_network.multilayer_perceptron import MLPClassifier

from DeepLearning import adapter

class ModelCreator(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def create(self):
        raise NotImplementedError('users must define \'create\' to use this base class')

class SimpleKerasRecurrentNNCreator(ModelCreator):

    def __init__(self, input_shape=None, numNeurouns=None, numOutputNeurons=None, activation='sigmoid', loss='categorical_crossentropy', optimizer='adam'):
        self.input_shape = input_shape
        self.numNeurons = numNeurouns
        self.numOutputNeurons = numOutputNeurons
        self.activation = activation
        self.loss = loss
        self.optimizer = optimizer

    def __build_model(self):
        model = Sequential()
        model.add(LSTM(self.numNeurons, input_shape=self.input_shape))
        model.add(Dense(self.numOutputNeurons, activation='sigmoid'))
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        return model

    def create(self):
        return adapter.KerasRecurrentNNAdapter(self.__build_model())


# class SimpleKerasCovolutionalNNCreator(ModelCreator):
#
#     def __build_model(self):
#
#
#     def create(self):

class SklearnNeuralNetowrk(ModelCreator):

    def __init__(self, solver="lbfgs", alpha=1e-5, hidden_layer_sizes=10, random_state=1):
        self.solver = solver
        self.alpha = alpha
        self.hidden_layer_sizes = hidden_layer_sizes
        self.random_state = random_state

    def create(self):
        model = MLPClassifier(solver=self.solver, alpha=self.alpha, hidden_layer_sizes=self.hidden_layer_sizes, random_state=self.random_state)
        return adapter.SklearnAdapter(model)