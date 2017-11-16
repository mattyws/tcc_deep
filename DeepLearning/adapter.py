import abc

import keras
import numpy as np
import itertools

from keras.models import load_model


class ModelAdapter(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fit(self, trainDocs, trainCats, epochs=0, batch_size=10):
        raise NotImplementedError('users must define \'fit\' to use this base class')

    @abc.abstractmethod
    def predict(self, testDocs, batch_size=10):
        raise NotImplementedError('users must define \'predict\' to use this base class')

    @abc.abstractmethod
    def predict_one(self, doc):
        raise NotImplementedError('users must define \'predict_one\' to use this base class')

    @abc.abstractmethod
    def fit_generator(self, data_generator, epochs=0, batch_size=10):
        raise NotImplementedError('users must define \'fit_generator\' to use this base class')

    @abc.abstractmethod
    def save(self, filename):
        raise NotImplementedError('users must define \'save\' to use this base class')

    @abc.abstractmethod
    def load(self, filename):
        raise NotImplementedError('users must define \'load\' to use this base class')

class KerasGeneratorAdapter(ModelAdapter):

    def __init__(self, model):
        self.model = model

    def fit(self, trainDocs, trainCats, epochs=1, batch_size=10):
        data = self.XYGenerator(trainDocs, trainCats)
        # for i in range(0, epochs):
        self.model.fit_generator(data, batch_size, epochs=epochs, initial_epoch=0, max_q_size=1)

    def fit_generator(self, data_generator, epochs=0, batch_size=10):
        self.model.fit_generator(data_generator, batch_size, epochs=epochs, initial_epoch=0, max_q_size=1)

    def predict(self, testDocs, batch_size=10):
        # data = self.XYGenerator(testDocs)
        pred = []
        result = self.model.predict_generator(testDocs, batch_size, max_q_size=1)
        for r in result:
            pred.append(np.argmax(r))
        return pred

    def predict_one(self, doc):
        result = self.model.predict(doc)
        return np.argmax(result)

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        return KerasGeneratorAdapter(load_model(filename))

    class XYGenerator(object):
        def __init__(self, train_docs, train_cats):
            self.train_docs = train_docs
            self.train_cats = train_cats

        def __iter__(self):
            return self

        def __next__(self):
            try:
                x = next(self.train_docs)
                y = next(self.train_cats)
                return x, y
            except:
                raise StopIteration()

        def next(self):
            return self.__next__()

class SklearnAdapter(ModelAdapter):

    def __init__(self, model):
        self.model = model

    def predict(self, testDocs, batch_size=10):
        pred = self.model.predict(testDocs)
        return pred

    def fit(self, trainDocs, trainCats, epochs=0, batch_size=10):
        self.model.fit(trainDocs, trainCats)

    def fit_generator(self, data_generator, epochs=0, batch_size=10):
        raise NotImplementedError('\'fit_generator\' not implemented in this class.')
