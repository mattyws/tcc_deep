import pymongo
import gridfs

client = pymongo.MongoClient()
patents_database = client.patents

metadata_collection = patents_database.documents_meta

total_documents = metadata_collection.find().count()

for i in range(2006, 2018):
    print(i)
    docs = metadata_collection.find({'filename': {'$regex': '.*'+str(i)+'.*'}})
    maxDocs = 66000
    aux = 0
    for doc in docs:
        docs_meta = patents_database.word2vec_docs
        # print(doc)
        docs_meta.insert_one({
            'filename': doc["filename"],
            'ipc_classes': doc["ipc_classes"]
        })
        aux += 1
        if aux > maxDocs:
            break
    print(aux)