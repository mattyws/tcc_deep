'''
====== Misc Routines ==========
Version 15-12-2016
Python 3.4.3
'''

from sklearn.cross_validation import *
import pandas as pd
from gensim.models.doc2vec import LabeledSentence



'''
input: list structure [[cat,text],[cat,text],...]
output:  pandas data frame
'''
class LstToDataFrame(object):
	def transform(corpus):
		dic = LstToDict(corpus).getDict()  #print(dic['data'])
		return(pd.DataFrame(dic, columns=['target','data'])) #print(docs.data)



class LstToDict(object):
	'''
	  input 
	'''
	def __init__(self, lstData):
		self.lst = lstData

	def getDict(self):
		dic = dict() 
		dic = {'target':[],'data':[]}
		for text in self.lst:
			dic['target'].append(text[0])
			dic['data'].append(text[1])
		return dic
	
	def getDictUnique(self):
		tokenDict = dict() 
		for text in self.lst:
			if text[0] in tokenDict:
				tokenDict[text[0]] = tokenDict[text[0]] + text[1]				
			else:
				tokenDict[text[0]] = text[1]
		return tokenDict


class LstToCatsAndDocsLst(object):
	def get(data):   # input[[cat,text],[cat,text]]
		cats = []
		docs = []
		for i in data:
			cats.append(i[0])
			docs.append(i[1])
		return(iter(cats), iter(docs))


class PrepareTrainAndTestData(object):
	
	def split(dataframe, percTrain, percTest, randomInt):
		(trainDocs, testDocs, trainCats, testCats) = train_test_split(dataframe.data, dataframe.target, test_size = (percTest/100), train_size= (percTrain/100), random_state = randomInt)
		#print("====Train======")
		#print(trainCats)
		#print(trainDocs)
		#print("====Test======")
		#print(testCats)
		#print(testDocs)
		return(trainDocs, testDocs, trainCats, testCats)

