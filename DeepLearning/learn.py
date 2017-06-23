import csv

from sklearn.model_selection import KFold
from gensim.models import doc2vec
from gensim.models.word2vec import Word2Vec
from keras.utils.np_utils import to_categorical
from sklearn.metrics.classification import accuracy_score, recall_score, precision_score, f1_score
import pandas as pd


class KFoldEvaluate(object):
    """
    The class responsable to evaluate a model using cross-validation method
    """
    def __init__(self):
        self.tableResults = None

    def initiate(self, modelFactory, x, y, folds=10, numClasses=None, epochs=1):
        """
        Initiate the cross-validation using a desired data

        :param modelFactory: a factory that builds a model
        :param x: the evaluation data
        :param y: the evaluation classes
        :param folds: how many folds for the cross-validation
        :param numClasses: the number of different classes that exists in the data
        :param toCategorical: if the classes have to be in a binary vector representation

        :type modelFactory: DeepLearning.model_creator.ModelCreator
        :type x: ndarray
        :type y: ndarray
        :type folds: int
        :type numClasses: int
        :type toCategorical: bool
        """
        self.folds = folds
        self.numClasses = numClasses
        self.epochs = epochs
        self.__evaluate(modelFactory, x, y)

    def get_results(self):
        """
        :return: The result from the cross-validation or None if no cross-validation was performed.
        """
        return self.tableResults

    def print_mean(self):
        df = pd.DataFrame(self.tableResults)
        filt = pd.pivot_table(df, values=['precision', 'recall', 'accuracy', 'f1'], index=['result'])
        print(filt)

    def save_result(self, filename):
        measures = ['precision', 'recall', 'accuracy', 'f1']
        df = pd.DataFrame(self.tableResults)
        filt = pd.pivot_table(df, values=['precision', 'recall', 'accuracy', 'f1'], index=['result'])
        print(" Results")
        print(filt)
        with open(filename, 'a') as f:
            writer = csv.writer(f, delimiter=',')
            for measure in measures:
                value = [measure]
                value.append(filt[measure])
                writer.writerow(value)
            writer.writerow([''])


    def __evaluate(self, modelFactory, x, y):

        """
        Perform the cross validation
        :param modelFactory: a factory that builds a model
        :param x: the evaluation data
        :param y: the evaluation classes
        """

        #Creating KFold
        kf = KFold(self.folds, shuffle=True, random_state=None)
        print("=============================" + str(self.folds) + "-fold Cross-Validation training and testing ============================= \n")
        i = 1
        # If the number of classes is not given, use the classes that we have
        if not self.numClasses:
            self.numClasses = len(set(y))
        # A list of results to be used to see how well the model is doing over the folds
        tableResults = []
        #Loop through the folds separation of data
        for trainIndex, testIndex in kf.split(x):
            # print(type(trainIndex))
            # Build a model adapter using a factory
            model = modelFactory.create()
            # A print to see if it is ok
            print(" ============== Fold ", i, "============")
            trainDocs, testDocs = x[trainIndex], x[testIndex]
            trainCats, testCats = y[trainIndex], y[testIndex]
            # If we want the categories to be represented as a binary array, here is were we do that
            #TODO: Categorical class error representation on valuating the classes returned by the model
            # Using the adapter to fit our model
            model.fit(trainDocs, trainCats, epochs=self.epochs, batch_size=len(trainIndex))
            # Predicting it
            pred = model.predict(testDocs, testCats)
            print(pred)
            # Getting the scores
            accuracy = accuracy_score(testCats, pred)
            recall = recall_score(testCats, pred, average='weighted')
            precision = precision_score(testCats, pred, average='weighted')
            f1 = f1_score(testCats, pred, average='weighted')
            #Appending it to the result table
            tableResults.append({'result': 'result', 'accuracy': accuracy, 'recall': recall, 'precision': precision, 'f1': f1})
            i += 1
        self.tableResults = tableResults

class Word2VecTrainer(object):
    """
    Perform training and save gensim word2vec
    """

    def __init__(self, min_count=2, size=200, workers=4, window=3, iter=10):
        self.min_count = min_count
        self.size = size
        self.workers = workers
        self.window = window
        self.iter = iter
        self.model = None

    def train(self, corpus):
        self.model = Word2Vec(corpus, min_count=self.min_count, size=self.size, workers=self.workers, window=self.window, iter=self.iter)

    def save(self, filename):
        self.model.save(filename)

    def get_model(self):
        return self.model

    def load_model(self, filename):
        return Word2Vec.load(filename)

class Doc2VecTrainer(object):
    """
    Perform training and save gensim doc2vec
    """

    def __init__(self, min_count=2, alpha=0.025, min_alpha=0.001, size=200, workers=4, window=3, iter=10):
        self.min_count = min_count
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.size = size
        self.workers = workers
        self.window = window
        self.iter = iter
        self.model = None

    def train(self, corpus):
        self.model = doc2vec.Doc2Vec(alpha=self.alpha, min_alpha=self.min_alpha, dm=0, hs=1, negative=0,
                                min_count=self.min_count, window=self.window, workers=self.workers, size=self.size)
        print("Building vocabulary")
        self.model.build_vocab(corpus)
        alpha_delta = (self.alpha - self.min_alpha) / self.iter

        for epoch in range(self.iter):
            print(str(epoch) + " epoch")
            self.model.alpha = self.alpha
            self.model.min_alpha = self.alpha  # fix the learning rate, no decay
            self.model.train(corpus)
            self.alpha -= alpha_delta

    def save(self, filename):
        self.model.save(filename)

    def get_model(self):
        return self.model

    def load_model(self, filename):
        return doc2vec.Doc2Vec.load(filename)
