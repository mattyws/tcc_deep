import pymongo

client = pymongo.MongoClient()
patents_database = client.patents

metadata_collection = patents_database.documents_meta
training_collection = patents_database.training_docs100
testing_collection = patents_database.testing_docs100
docs = metadata_collection.find()


main_ipc = set()
for doc in docs:
    if len(doc['ipc_classes']) > 0 :
        main_ipc.add(doc['ipc_classes'][0])
    else:
        print(doc['filename'])
print("Creating  database")

# classes = 0
# for ipc_class in main_ipc:
#     # if classes > 50:
#     #     break
#     print(ipc_class)
#     i = 100
#     while i > 0:
#         try:
#             doc = docs.next()
#             training_collection.insert_one({
#                 'filename': doc["filename"],
#                 'ipc_classes': doc["ipc_classes"],
#             })
#         except:
#             break
#         i -= 1
#     classes += 1

# print(len(main_ipc))

for ipc_class in main_ipc:
    print(ipc_class)
    docs = metadata_collection.find({"ipc_classes.0" : ipc_class})
    i = 40
    while i > 0:
        try:
            doc = docs.next()
            if not training_collection.find({'filename': doc['filename']}, {'_id': 1}).count() > 0:
                print(doc['filename'] + " not in")
                testing_collection.insert_one({
                    'filename': doc["filename"],
                    'ipc_classes': doc["ipc_classes"],
                })
                i -= 1
        except:
            break
    # classes += 1