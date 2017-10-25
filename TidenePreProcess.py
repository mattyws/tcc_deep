'''
Version 15-12-2016
Python 3.4.3
'''

'''
Preprocess strings
'''

from nltk.tokenize import *


class TokenizeFromStreamFunction(object):
	'''
	Constructor param = tokenizer
	Input: stream
	Output: list = []
	'''
	def __init__(self, tokenizer):
		self.tokenizer = tokenizer
		
	def get(self, stream):
		return(self.tokenizer.tokenize(stream))


class Tokenize(object):

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)

class TokenizeFromList(object):
	'''
	Constructor param = tokenizer, data
	Input: [text,text,..]
	Output: iter to a tokenized list = [[token1, token2, ..],[token1, token2,...], ...]
	'''
	def __init__(self, tokenizer, lstData):
		self.lst = lstData
		self.tokenizer = tokenizer
		
	def __iter__(self):
		for text in self.lst:
			t = self.tokenizer.tokenize(text)
			if len(t) > 300 :
				yield t[0:300]
			else:
				yield t




class TokenizeAndStemFromStream(object):
	'''
	Constructor param = tokenizer, stream
	Input: stream
	Output: list = []
	'''
	def __init__(self, tokenizer, stemmer):
		self.tokenizer = tokenizer
		self.stemmer = stemmer
	
	def get(self,stream):
		return(StemTokensFromSimpleList(self.stemmer).stm(self.tokenizer.tokenize(stream)))


class CleanStopWords(object):
	'''
	Constructor param = stopset, data
	Input: lst
	Output: iter to a list = [[text],[text]]
	'''
	
	def __init__(self, stopSet):
		self.stopSet = stopSet
	
	def clean(self, text):
		return [w for w in text if not w.lower() in self.stopSet]


class CleanStopWordsOld(object):
	'''
	Constructor param = stopset, data
	Input: [[cat,text],[cat,text]]
	Output: iter to a tokenized list = [[cat,text],[cat,text]]
	'''
	
	def __init__(self, stopSet, lstData):
		self.lst = lstData
		self.stopSet = stopSet
	
	def __iter__(self):
		data = [] 
		for text in self.lst: 
			yield [w for w in text[1] if not w in self.stopSet]
			


class StemTokensFromSimpleList(object):
	'''
	Constructor param = stemmer, data
	Input: []
	Output: iter to a tokenized list = []
	'''	
	def __init__(self, stemmer):
		self.stemmer = stemmer
	
	def stm(self,lst):
		for text in lst:
			return [self.stemmer.stem(w) for w in lst]


class StemTokensFromStructCatList(object):
	'''
	Constructor param = stemmer, data
	Input: [[cat,text],[cat,text]]
	Output: iter to a tokenized list = [[cat,text],[cat,text]]
	'''	
	def __init__(self, stemmer, lstData):
		self.lst = lstData
		self.stemmer = stemmer
	
	def __iter__(self):
		data = [] 
		for text in self.lst:
			yield [text[0],[self.stemmer.stem(w) for w in text[1]]]

