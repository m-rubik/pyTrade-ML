"""!
All functions related to the generation and storing of AlphaVantage keys.

# https://www.alphavantage.co/documentation/
# https://github.com/RomelTorres/alpha_vantage
# https://alpha-vantage.readthedocs.io/en/latest/genindex.html
# https://alpha-vantage.readthedocs.io/en/latest/source/alpha_vantage.html#module-alpha_vantage.timeseries
"""

import os
from src.utilities.pickle_utilities import import_object, export_object


def add_key(key):
    """!
    Add a new key to the list.
    """
    key_list = import_object("KEYS")
    if key_list:
        if key in key_list:
            print("Key", key, "already stored.")
            flag = 0
        else:
            key_list.append(key)
            flag = export_object("KEYS", key_list)
    else:
        print("Generating new key list.")
        key_list = list()
        key_list.append(key)
        flag = export_object("KEYS", key_list)
    return flag


def load_key(index=0):
    """!
    Load a key in the list by index
    """
    key_list = import_object("KEYS")
    if key_list:
        print(key_list)
        return key_list[index]
    else:
        return None


def remove_key(key):
    key_list = import_object("KEYS")
    if key_list:
        if key in key_list:
            key_list.remove(key)
            flag = export_object("KEYS", key_list)
        else:
            print("Key", key, "not found.")
            flag = 0
    else:
        flag = 1
    return flag


def unittest():
    add_key("test")
    print(load_key())
    remove_key('test')


if __name__ == "__main__":
    # add_key("XXXXXXXXXXXXXXXXX")
    print(load_key())
