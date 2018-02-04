import abc

from keras.layers.convolutional import Conv2D, Conv1D
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.pooling import MaxPool2D, MaxPool1D, AveragePooling1D
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.optimizers import SGD
from sklearn.neural_network.multilayer_perceptron import MLPClassifier

from DeepLearning import adapter

class ModelCreator(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def create(self):
        raise NotImplementedError('users must define \'create\' to use this base class')

class SimpleKerasRecurrentNNCreator(ModelCreator):

    def __init__(self, input_shape=None, numNeurouns=None, numOutputNeurons=None, activation='sigmoid', loss='categorical_crossentropy', optimizer='adam', use_dropout=False, dropout=0.5):
        self.input_shape = input_shape
        self.numNeurons = numNeurouns
        self.numOutputNeurons = numOutputNeurons
        self.activation = activation
        self.loss = loss
        self.optimizer = optimizer
        self.use_dropout = use_dropout
        self.dropout = dropout

    def __build_model(self):
        model = Sequential()
        model.add(LSTM(self.numNeurons, input_shape=self.input_shape))
        if self.use_dropout:
            model.add(Dropout(self.dropout))
        model.add(Dense(self.numOutputNeurons, activation='sigmoid'))
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        return model

    def create(self):
        return adapter.KerasGeneratorAdapter(self.__build_model())

class MultilayerKerasRecurrentNNCreator(ModelCreator):
    def __init__(self, input_shape=None, numNeurouns=None, numOutputNeurons=None, activation='sigmoid', loss='categorical_crossentropy', optimizer='adam', layers=2, use_dropout=False, dropout=0.5):
        self.input_shape = input_shape
        self.numNeurons = numNeurouns
        self.numOutputNeurons = numOutputNeurons
        self.activation = activation
        self.loss = loss
        self.optimizer = optimizer
        self.layers = layers
        self.use_dropout = use_dropout
        self.dropout = dropout

    def __build_model(self):
        model = Sequential()
        for i in range(0, self.layers-1):
            model.add(LSTM(self.numNeurons, input_shape=self.input_shape, return_sequences=True))
            if self.use_dropout:
                model.add(Dropout(self.dropout))
        model.add(LSTM(self.numNeurons, input_shape=self.input_shape))
        if self.use_dropout:
            model.add(Dropout(self.dropout))
        model.add(Dense(self.numOutputNeurons, activation='sigmoid'))
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        return model

    def create(self):
        return adapter.KerasGeneratorAdapter(self.__build_model())

class KerasCovolutionalNNCreator(ModelCreator):

    #TODO: Finish this class
    def __init__(self, input_shape=None, loss='mse', optimizer='sgd'):
        self.input_shape = input_shape
        self.loss = loss
        self.optimizer = optimizer

    def __build_model(self):
        model = Sequential()
        model.add(Conv1D(128, kernel_size=3, activation='relu', padding='same', input_shape=self.input_shape))
        model.add(AveragePooling1D(pool_size=4, padding="same"))
        model.add(Dropout(0.5))
        # model.add(Conv1D(128, kernel_size=3, activation='elu', padding='same'))
        # model.add(AveragePooling1D(pool_size=1, padding="same"))
        # model.add(Dropout(0.5))
        # model.add(Conv1D(16, kernel_size=5, activation='elu', padding='same'))
        # model.add(MaxPool1D(pool_size=1, padding="same"))
        # model.add(Conv1D(16, kernel_size=5, activation='elu', padding='same'))
        # model.add(Dropout(0.25))

        # model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Dropout(0.5))
        model.add(Dense(8, activation='tanh'))
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        return model

    def create(self):
        return adapter.KerasGeneratorAdapter(self.__build_model())

class KerasMultilayerPerceptron(ModelCreator):

    def __init__(self, num_class=8, input_dim=200, layers=1, hidden_units=[20], use_dropout=True, dropout=0.5):
        if len(hidden_units) != layers:
            raise ValueError("The hidden_units must have the size of the number of layers.")
        self.input_dim = input_dim
        self.layers = layers
        self.hidden_units = hidden_units
        self.use_dropout = use_dropout
        self.dropout = dropout
        self.num_class = num_class

    def __build_model(self):
        model = Sequential()
        i = 0
        for i in range(self.layers):
            model.add(Dense(self.hidden_units[i], activation='relu', input_dim=self.input_dim))
            if self.use_dropout:
                model.add(Dropout(self.dropout))
        model.add(Dense(self.num_class, activation='softmax'))
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])
        return model

    def create(self):
        return adapter.KerasGeneratorAdapter(self.__build_model())

# class SimpleKerasCovolutionalNNCreator(ModelCreator):
#
#     def __build_model(self):
#
#
#     def create(self):

class SklearnNeuralNetwork(ModelCreator):

    def __init__(self, solver="lbfgs", alpha=1e-5, hidden_layer_sizes=10, random_state=1):
        self.solver = solver
        self.alpha = alpha
        self.hidden_layer_sizes = hidden_layer_sizes
        self.random_state = random_state

    def create(self):
        model = MLPClassifier(solver=self.solver, alpha=self.alpha, hidden_layer_sizes=self.hidden_layer_sizes, random_state=self.random_state)
        return adapter.SklearnAdapter(model)
