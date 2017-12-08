import os
import pymongo

path = 'base100'

client = pymongo.MongoClient()
patents_database = client.patents
metadata_collection = patents_database.documents_meta
training_collection = patents_database.training_old

all_files = []
number = 0
for dirName, subdirList, fileList in os.walk(path):
    for file in fileList:
        all_files.append(file)

for doc in all_files:
    patent = metadata_collection.find_one({"filename" : doc})
    if not training_collection.find({'filename': patent['filename']}, {'_id': 1}).count() > 0:
        print(doc)
        training_collection.insert_one({
            'filename': patent["filename"],
            'ipc_classes': patent["ipc_classes"],
        })
    else:
        print("Already add "+doc)
