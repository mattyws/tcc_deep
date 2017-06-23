'''
Extract Features
Version 15-12-2016
Python 3.4.3
'''

from nltk.tokenize import *
from sklearn.feature_extraction.text import *
from collections import defaultdict
import numpy as np

import TidenePreProcess

#http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

class TfidfSklearn(object):
	def getVec(tokenizer,stopSet):
		tokenFunction = TidenePreProcess.TokenizeFromStreamFunction(tokenizer).get  # tokenize
		vectorizer = TfidfVectorizer(tokenizer= tokenFunction, ngram_range=(1,3), min_df = 0, stop_words=stopSet)
		transformer = TfidfTransformer()
		return(vectorizer, transformer)
		

		
class TfidfEmbeddingVectorizer(object):
	
	def __init__(self,word2vec,tokenizer, stopSet):
		self.word2vec = word2vec
		self.word2weight = None
		self.dim = len(word2vec)
		self.tokenizer = tokenizer
		self.stopSet = stopSet
	
	def fit(self, X, y):
		tokenFunction = TidenePreProcess.TokenizeFromStreamFunction(self.tokenizer).get  # tokenize
		tfidf = TfidfVectorizer(tokenizer= tokenFunction, ngram_range=(1,3), min_df = 0, stop_words=self.stopSet)
		tfidf.fit(X,y)
		max_idf = max(tfidf.idf_)
		self.word2weight = defaultdict(lambda: max_idf,[(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
		return self
	
	def transform(self, X):
		return np.array([np.mean([self.word2vec[w] * self.word2weight[w] for w in words if w in self.word2vec] or [np.zeros(self.dim)], axis=0) for words in X])
		
