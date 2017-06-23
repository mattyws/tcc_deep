'''
Version 15-12-2016
Python 3.4.3
'''

'''
Normalize strings
# Input: [[cat,text],[cat,text],...]
Output: 
  iter to a normalized list = [[cat,text],[cat,text],...]
'''

import unicodedata
import nltk
from nltk.tokenize import *

class TextNormalizePort(object):

	def __init__(self, lstData):
		print("======================= Normalizing Data ===============")
		self.lst = lstData

	def __iter__(self):
		data = [] 
		for text in self.lst: 
			# for texts in portuguese
			try:
				textNorm = unicodedata.normalize('NFKD', text[1]).encode('ASCII','ignore').decode('UTF-8')
			except ValueError:
				textNorm = unicodedata.normalize('NFKD',text[1]).encode('UTF-8','ignore').decode()
			yield [text[0],textNorm]

			
	


