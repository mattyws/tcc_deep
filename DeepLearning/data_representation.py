import string

from keras.utils.np_utils import to_categorical
import numpy as np
import pickle


class Word2VecEmbeddingCreator(object):

    """
    A class that transforms a data into their representation of word embedding
    It uses a trained word2vec model model to build a 3 dimentional vector representation of the document.
     The first dimension represents the document, the second dimension represents the word and the third dimension is the word embedding array
    """

    def __init__(self, word2vecModel, maxWords=300, embeddingSize=200):
        self.word2vecModel = word2vecModel
        self.maxWords = maxWords
        self.embeddingSize = embeddingSize
        self.num_docs = 0


    def create_x(self, d, max_words=0):
        """
        Transform a doc2vec.TaggedDocument words into a third dimensional array
        :param d: doc2vec.TaggedDocument of a document
        :param max_words: the max number of words to put into the 3 dimensional array
        :return: the 3 dimensional array representing the content of the document
        """
        if max_words == 0:
            max_words = self.maxWords
        x = np.zeros(shape=(1, max_words, self.embeddingSize), dtype='float')
        for helper2, w in enumerate(d.words):
            if helper2 >= max_words:
                break
            try:
                x[0, helper2] = self.word2vecModel[w]
            except:
                x[0, helper2] = np.zeros(shape=self.embeddingSize)
        return x

    def create_x_text(self, text, max_words=0):
        """
        Transform a tokenized text into a 3 dimensional array with the word2vec model
        :param text: the tokenized text
        :param max_words: the max number of words to put into the 3 dimensional array
        :return: the 3 dimensional array representing the content of the tokenized text
        """
        if max_words == 0:
            max_words = self.maxWords
        x = np.zeros(shape=(1, max_words, self.embeddingSize), dtype='int32')
        for helper2, w in enumerate(text):
            if helper2 >= max_words:
                break
            try:
                x[0, helper2] = self.word2vecModel[w]
            except:
                x[0, helper2] = np.zeros(shape=self.embeddingSize)
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
