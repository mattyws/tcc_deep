import os
import pymongo
from random import shuffle


path = 'base100'

client = pymongo.MongoClient()
patents_database = client.patents
collection = patents_database.training_docs100
shuffled_collection = patents_database.shuffled_training_docs100

all_documents = collection.find({}, {"filename":1})
shuffled = []
for doc in all_documents:
    shuffled.append(doc["filename"])
shuffle(shuffled)
for doc in shuffled:
    patent = collection.find_one({"filename":doc})
    shuffled_collection.insert_one(patent)
