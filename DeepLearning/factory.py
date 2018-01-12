import abc

from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.models import Sequential

from DeepLearning import model_creators

class ModelCreatorFactory(object):

    def __init__(self):
        self.MODEL_TYPES = dict()

    def create(self, model_type, **kwargs):
        return self.MODEL_TYPES[model_type](**kwargs)

    def add_model(self, name, model_class):
        self.MODEL_TYPES[name] = model_class


factory = ModelCreatorFactory()
factory.add_model('SimpleKerasRecurrentNN', model_creators.SimpleKerasRecurrentNNCreator)
factory.add_model('SkleanNeuralNetwork', model_creators.SklearnNeuralNetwork)
factory.add_model('MultilayerKerasRecurrentNN', model_creators.MultilayerKerasRecurrentNNCreator)
factory.add_model('KerasCovolutionalNetwork', model_creators.KerasCovolutionalNNCreator)
