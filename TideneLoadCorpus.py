'''
Version 15-12-2016
Python 3.4.3
'''

'''
# load data from path
# input: path
# output: iter to list = [[cat,text],[cat,text]
'''


import os

class GetCorpusFromPath(object):
	def __init__(self, path):
		print("======================= Loading Corpus ===============")
		self.path = path
	
	def __iter__(self):
		import os
		for dirName, subdirList, fileList in os.walk(self.path):
			for fname in fileList:
				cat = dirName.split("/")[-1]
				if (cat == ""):
					p = dirName + fname
					cat = fname.split(".")[0]
				else:
					p = dirName+"/"+fname
				try:
					# Reading utf-8 file
					stream = open(p, encoding="UTF-8").read().replace("\n"," ").lower()
				except ValueError:
					# if error Read as ISO-8859-15 file
					stream = open(p, encoding="ISO-8859-15").read().replace("\n"," ").lower()
				yield [cat,stream]
				
				
