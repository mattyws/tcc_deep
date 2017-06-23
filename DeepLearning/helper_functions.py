import numpy as np

def classMap(keys):
    i = 0
    map = dict()
    keys.sort()
    for k in keys:
        map[k] = i
        map[i] = k
        i+=1
    return map

def separate_class(keys, level):
    data = []
    for key in keys:
        key_subclasses = dict()
        for lKey in level.keys():
            if lKey.startswith(key):
                key_subclasses[lKey] = level[lKey]
        data.append(key_subclasses)
    return data

def get_pathname(keys, dataP):
    if not keys:
        return 'base'
    for k in keys:
        for data_key in dataP.keys():
            if data_key.startswith(k):
                return k

def change_to_number(y, class_map):
    number_classes = np.zeros(shape=len(y))
    for num, label in enumerate(y):
        number_classes[num] = class_map[label]
    return number_classes

def dictionary_to_list(path_dict):
    path_list = []
    for key in path_dict:
        for path in path_dict[key]:
            path_list.append((key, path))
    print(len(path_list))
    # print(path_list)
    return path_list