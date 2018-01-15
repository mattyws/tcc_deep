import os
import pymongo
from random import shuffle


path = 'base100'

client = pymongo.MongoClient()
patents_database = client.patents
collection = patents_database.testing_embedding_old
shuffled_collection = patents_database.shuffled_testing_embedding_old

all_documents = collection.find({}, {"filename":1})
shuffled = []
for doc in all_documents:
    shuffled.append(doc["filename"])
shuffle(shuffled)
i=0
for doc in shuffled:
    if i%1000 == 0:
        print(str(i) + ' ' + doc)
    patent = collection.find_one({"filename":doc})
    shuffled_collection.insert_one(patent)
    i+=1
