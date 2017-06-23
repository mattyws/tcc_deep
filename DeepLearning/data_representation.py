import string

from keras.utils.np_utils import to_categorical
import numpy as np
import pickle


class Word2VecEmbeddingCreator(object):

    """
    A class that transforms a data into their representation of word embedding
    It uses a trained word2vec model and doc2vec model to build a 3 dimentional vector of the representations.
     The first dimention represents the document, the second dimension represents the word and the third dimension is the word embedding array
    """

    def __init__(self, word2vecModel, maxWords=300, embeddingSize=200):
        self.word2vecModel = word2vecModel
        self.maxWords = maxWords
        self.embeddingSize = embeddingSize
        self.num_docs = 0

    def create_x(self, d, max_words=0):
        if max_words == 0:
            max_words = self.maxWords
        x = np.zeros(shape=(1, max_words, self.embeddingSize), dtype='int32')
        for helper2, w in enumerate(d.words):
            if helper2 >= max_words:
                break
            try:
                x[0, helper2] = self.word2vecModel[w]
            except:
                x[0, helper2] = np.zeros(shape=self.embeddingSize)
        # print(x)
        # with open('outfile', 'wb') as fp:
        #     pickle.dump(x, fp)
        # print("dump")
        # with open('outfile', 'rb') as fp:
        #     itemlist = pickle.load(fp)
        #     print(itemlist)
        return x


class LabelsCreator(object):
    def __init__(self, class_map, num_classes, labels_to_categorical=False):
        self.class_map = class_map
        self.labels_to_categorical = labels_to_categorical
        self.num_classes = num_classes
        self.num_docs = 0

    def create_y(self, d):
        if self.labels_to_categorical:
            return to_categorical(self.class_map[d.tags[0]], self.num_classes)
        else:
            return d.tags[0]
