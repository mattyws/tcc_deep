import os

from keras.utils.np_utils import to_categorical
from pymongo import MongoClient
from queue import Queue
from random import shuffle
from threading import Thread
import pymongo
import gridfs

from gensim.models import doc2vec
import numpy as np
import pickle

import TidenePreProcess


class GetFileContent(object):

    def get_content(self, path):
        try:
            # Reading utf-8 file
            with open(path, encoding="UTF-8") as f:
                stream = f.read().replace("\n", " ").lower()
        except ValueError:
            # if error Read as ISO-8859-15 file
            with open(path, encoding="ISO-8859-15") as f:
                stream = f.read().replace("\n", " ").lower()
        return stream


class GetFilesFromPath(object):
    """
    A generator that holds files content from a dictionary of lists of files paths, while interating over them.
    Each key of the dictionary will be considerated as the content class.
    """
    def __init__(self, path_list):
        self.path_list = path_list
        # shuffle(self.path_list)
        self.rebot()

    def next(self):
        return self.__next__()

    def rebot(self):
        # print("Rebot corpus")
        self._pos = -1

    def __iter__(self):
        self.rebot()
        return self

    def __next__(self):
        # print(" Pos: " + str(self._pos))
        if self._pos >= len(self.path_list)-1:
            raise StopIteration
        else:
            self._pos += 1
            return [self.path_list[self._pos][0], GetFileContent().get_content(self.path_list[self._pos][1])]

    def __getitem__(self, item):
        if isinstance(item, int):
            yield [self.path_list[item][0], GetFileContent().get_content(self.path_list[item][1])]
            raise StopIteration
        elif isinstance(item, slice):
            raise NotImplemented("Indexing with slices not implemented")
        elif isinstance(item, tuple) or isinstance(item, np.ndarray):
            for i in item:
                yield [self.path_list[i][0], GetFileContent().get_content(self.path_list[i][1])]
            raise StopIteration


    def __len__(self):
        return len(self.path_list)



class LoadTextCorpus(object):
    """
    Pre-process the corpus
    """
    def __init__(self, path_dict, tokenizer, stop_set=None, stemmer=None):
        self.path_dict = path_dict
        self.tokenizer = tokenizer
        self.stop_set = stop_set
        self.stemmer = stemmer
        self.corpus = GetFilesFromPath(self.path_dict)

    def next(self):
        return self.__next__()

    def rebot(self):
        # print("Rebooting")
        self.corpus = GetFilesFromPath(self.path_dict)

    def __iter__(self):
        self.corpus = GetFilesFromPath(self.path_dict)
        return self

    def __preprocess(self, text):
        text = TidenePreProcess.TokenizeFromList(self.tokenizer, [text])
        if self.stop_set is not None:
            text = TidenePreProcess.CleanStopWords(self.stop_set).clean(text)
        for t in text:
            if self.stemmer is not None:
                t = [self.stemmer.stem(word) for word in t]
            return t

    def __next__(self):
        try:

            data = next(self.corpus)
            t = self.__preprocess(data[1])
            return doc2vec.TaggedDocument(t, [data[0]])
        except StopIteration:
            self.rebot()
            raise StopIteration

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, item):
        for data in self.corpus[item]:
            t = self.__preprocess(data[1])
            yield doc2vec.TaggedDocument(t, data[0])
        raise StopIteration()




class LoadFilesContent(object):
    """
    Pre-process the corpus
    """
    def __init__(self, path_dict, tokenizer, stop_set=None, stemmer=None):
        self.path_dict = path_dict
        self.tokenizer = tokenizer
        self.stop_set = stop_set
        self.stemmer = stemmer

    def __iter__(self):
        self.corpus = GetFilesFromPath(self.path_dict)
        for data in self.corpus:
            text = TidenePreProcess.Tokenize(self.tokenizer).clean(data[1])
            if self.stop_set is not None:
                text = TidenePreProcess.CleanStopWords(self.stop_set).clean(text)
            for t in text:
                if self.stemmer is not None:
                    t = [self.stemmer.stem(word) for word in t]
                yield t

    def __len__(self):
        return len(self.path_dict)

class XGenerator(object):

    def __init__(self, transformer, data, loop_forever=True):
        self.transformer = transformer
        self.data = data
        self.loop_forever = loop_forever

    def __iter__(self):
        self.data.rebot()
        return self

    def __next__(self):
        try :
            d = next(self.data)
            return self.transformer.create_x(d)
        except StopIteration:
            if self.loop_forever:
                self.data.rebot()
            else:
                raise StopIteration

    def __len__(self):
        return len(self.data)


class YGenerator(object):

    def __init__(self, transformer, data, loop_forever=True):
        self.transformer = transformer
        self.data = data
        self.loop_forever = loop_forever

    def __iter__(self):
        self.data.rebot()
        return self

    def __next__(self):
        try :
            d = next(self.data)
            return self.transformer.create_y(d)
        except StopIteration:
            if self.loop_forever:
                self.data.rebot()
            else:
                raise StopIteration

    def __len__(self):
        return len(self.data)


class HierarchicalStructureDatabase(object):
    """
    Class that loads the patents based on their classification. Initiate the class using the Subclass classification path.
    The data must be in a hierarchcal structure of classes.You can either iter this class or call their structure separated
    """
    def __init__(self, path):
        self.path_list = self.__get_files_paths(path)

    def __get_files_paths(self, path):
        all_files = []
        number = 0
        for dirName, subdirList, fileList in os.walk(path):
            for file in fileList:
                all_files.append(dirName + '/' + file)
        return all_files

    '''

    '''
    def sections(self):
        """
        Get a dict of files, where each key is a section in the patent IPC classification (A-H)
        :return: a dict of files, where each key is a section in the patent IPC classification (A-H)
        """
        section_dict = dict()
        for path in  self.path_list:
            if not path.split('/')[-4] in section_dict.keys():
                section_dict[path.split('/')[-4]] = []
            section_dict[path.split('/')[-4]].append(path)
        return section_dict

    def classes(self):
        """
        Get a dic of files, where each key is a class in the patent IPC classification (The patent's section + two numbers)
        :return: a dic of files, where each key is a class in the patent IPC classification (The patent's section + two numbers)
        """
        class_dict = dict()
        for path in self.path_list:
            if not path.split('/')[-3] in class_dict.keys():
                class_dict[path.split('/')[-3]] = []
            class_dict[path.split('/')[-3]].append(path)
        return class_dict

    def subclasses(self):
        """
        Get a dict of files, where each key is a subclass in the patent IPC classification (The patent's section + the patent's
        class + a letter (A-Z)
        :return: a dict of files, where each key is a subclass in the patent IPC classification (The patent's section + the patent's class + a letter (A-Z)
        """
        subclass_dict = dict()
        for path in self.path_list:
            if not path.split('/')[-2] in subclass_dict.keys():
                subclass_dict[path.split('/')[-2]] = []
            subclass_dict[path.split('/')[-2]].append(path)
        return subclass_dict

    def __iter__(self):
        for level in  [self.sections(), self.classes(), self.subclasses()]:
            yield level


class FlatStructureDatabase(object):

    """
    Class that loads the patents based on their classification. Initiate the class using the Subclass classification path.
    The data must be in a flat structure of classes. You can either iter this class or call their structure separated
    """
    def __init__(self, path):
        self.path_list = self.__get_files_paths(path)

    def __get_files_paths(self, path):
        all_files = []
        number = 0
        for dirName, subdirList, fileList in os.walk(path):
            for file in fileList:
                all_files.append(dirName + '/' + file)
        return all_files

    def sections(self):
        """
        Get a dict of files, where each key is a section in the patent IPC classification (A-H)
        :return: a dict of files, where each key is a section in the patent IPC classification (A-H)
        """
        section_dict = dict()
        for path in  self.path_list:
            if not path.split('/')[-2][0] in section_dict.keys():
                section_dict[path.split('/')[-2][0]] = []
            section_dict[path.split('/')[-2][0]].append(path)
        return section_dict

    def classes(self):
        """
        Get a dic of files, where each key is a class in the patent IPC classification (The patent's section + two numbers)
        :return: a dic of files, where each key is a class in the patent IPC classification (The patent's section + two numbers)
        """
        class_dict = dict()
        for path in self.path_list:
            if not path.split('/')[-2][0:3] in class_dict.keys():
                class_dict[path.split('/')[-2][0:3]] = []
            class_dict[path.split('/')[-2][0:3]].append(path)
        return class_dict

    def subclasses(self):
        """
        Get a dict of files, where each key is a subclass in the patent IPC classification (The patent's section + the patent's
        class + a letter (A-Z)
        :return: a dict of files, where each key is a subclass in the patent IPC classification (The patent's section + the patent's class + a letter (A-Z)
        """
        subclass_dict = dict()
        for path in self.path_list:
            if not path.split('/')[-2] in subclass_dict.keys():
                subclass_dict[path.split('/')[-2]] = []
            subclass_dict[path.split('/')[-2]].append(path)
        return subclass_dict

    def __iter__(self):
        for level in  [self.sections(), self.classes(), self.subclasses()]:
            yield level


class ObjectDatabaseSaver(object):

    def __init__(self, filename):
        self.filename = filename
        if os.path.exists(filename):
            os.remove(filename)

    def save(self, d):
        with open(self.filename, "ab") as file:
            pickle.dump(d, file)

class ObjectDatabaseReader(object):

    # class ObjectDatabaseProducer(Thread):
    #
    #     def __init__(self, file, queue, end_of_file):
    #         Thread.__init__(self)
    #         self.file = file
    #         self.queue = queue
    #         self.end_of_file = end_of_file
    #
    #     def run(self):
    #         while not self.queue.full():
    #             try :
    #                 self.queue.put(pickle.load(self.file))
    #             except EOFError:
    #                 print("EOF")
    #                 self.end_of_file = True
    #                 break


    def __init__(self, filename, serve_forever=False, batch_size=10):
        self.file = open(filename, "rb")
        self.filename = filename
        self.serve_forever = serve_forever
        self.batch_size = batch_size
        self.queue = Queue(maxsize=batch_size)
        self.end_of_file = False

    def __reboot(self):
        self.file = open(self.filename, "rb")
        self.end_of_file = False
        self.__fill_queue()

    def __fill_queue(self):
        while not self.queue.full():
            try:
                self.queue.put(pickle.load(self.file), timeout=200)
            except EOFError:
                # print("EOF")
                self.end_of_file = True
                break

    def __iter__(self):
        self.__fill_queue()
        return self

    def __next__(self):
        if self.queue.empty():
            self.__fill_queue()
            # print(self.end_of_file, self.serve_forever)
            if self.end_of_file:
                if self.serve_forever:
                    self.__reboot()
                else:
                    raise StopIteration
        return self.queue.get()

    # def __init__(self, filename, serve_forever=False):
    #     self.filename = filename
    #     self.serve_forever = serve_forever
    #     self.file = open(self.filename, "rb")
    #
    # def __iter__(self):
    #     self.file = open(self.filename, "rb")
    #     return self
    #
    # def __next__(self):
    #     try:
    #         d = pickle.load(self.file)
    #         return d
    #     except EOFError:
    #         if self.serve_forever:
    #             self.file = open(self.filename, "rb")
    #         else:
    #             raise StopIteration


class MongoLoadDocumentMeta(object):

    def __init__(self, database):
        self.client = MongoClient()
        # client = MongoClient()
        self.database = self.client[database]
        # fs = gridfs.GridFS(patents_database)
        # doc = fs.get_last_version(filename="US20140250555A1-20140911")
        #
        # print(doc.read())

    def get_all_meta(self, collection):
        return self.database[collection].find().batch_size(2) #.sort([('filename', 1)])

    def get_meta_by_section(self, collection, section):
        return self.database[collection].find({"ipc_classes.0":{"$regex":"^"+section+""}})




class MongoLoadDocumentData(object):

    def __init__(self, database, documents_meta, clean_text=False, tokenizer=None, stop_set=None, stemmer=None, abstract=False, description=False, claims=False, doc2vec_doc=False):
        self.client = MongoClient()
        self.database = self.client[database]
        self.documents_meta = documents_meta
        self.documents_meta.rewind()

        self.doc2vec_doc = doc2vec_doc
        self.abstract = abstract
        self.description = description
        self.claims = claims
        self.clean_text = clean_text
        self.tokenizer = tokenizer
        self.stop_set = stop_set
        self.stemmer = stemmer

        self.abstracts = gridfs.GridFS(self.database, collection="abstracts")
        self.claims_grid = gridfs.GridFS(self.database, collection="claims")
        self.descriptions = gridfs.GridFS(self.database, collection="descriptions")

    def clean(self, text):
        text = TidenePreProcess.Tokenize(self.tokenizer).tokenize(text)
        if self.stop_set is not None:
            text = TidenePreProcess.CleanStopWords(self.stop_set).clean(text)
        if self.stemmer is not None:
            text = [self.stemmer.stem(word) for word in text]
        return text

    def get_file_content(self, filename):
        document = dict()
        if not (self.abstract or self.description or self.claims):
            return None
        if self.abstract:
            document["abstract"] = self.abstracts.get_last_version(filename).read().decode()
        if self.description:
            document["description"] = self.descriptions.get_last_version(filename).read().decode()
        if self.claims:
            document["claims"] = self.claims_grid.get_last_version(filename).read().decode()
        return document

    def __iter__(self):
        for document in self.documents_meta:
            content = self.get_file_content(document['filename'])
            result = ''
            for key in content.keys():
                if self.clean_text and self.tokenizer is not None:
                    result = self.clean(content[key])
                else:
                    result = content[key]
                if self.doc2vec_doc:
                    result = doc2vec.TaggedDocument(document['ipc_classes'][0], result)
                yield result
        self.documents_meta.rewind()

class PatentsCollectionManagement(object):

    def get_all(self, collection):
        return collection.find()

    def get_by_section(self, collection, section):
        collection.find({"ipc_classes.0" : {'$regex' : '/^'+section+'/'}})


class MongoDBMetaEmbeddingGenerator(object):

    def __init__(self, mongo_iterator, hierarchy_level, class_map, num_classes, serve_forever=False):
        self.mongo_iterator = mongo_iterator
        self.hierarchy_level = hierarchy_level
        self.class_map = class_map
        self.num_classes = num_classes
        self.serve_forever = serve_forever

    def __iter__(self):
        return self

    def __next__(self):
        try:
            document = self.mongo_iterator.next()
            x = pickle.loads(document['embedding'])
            y=None
            if self.hierarchy_level == "section":
                y = document['ipc_classes'][0][0]
            if self.hierarchy_level == "class":
                y = document['ipc_classes'][0][0:3]
            if self.hierarchy_level == "subclass":
                y = document['ipc_classes'][0]
            return x, to_categorical(self.class_map[y], self.num_classes)
        except:
            if self.serve_forever:
                self.mongo_iterator.rewind()
                return self.__next__()
            else:
                raise StopIteration()

    def next(self):
        return self.__next__()
