"""!
Contains all pickling functions.
"""

import pickle
from pathlib import Path


def import_object(filename):
    """!
    Import a pickled object into python object.
    @param filename: Path of the pickled object
    """

    if Path(filename).is_file():
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            return data
    else:
        return None

def export_object(filename, data):
    """!
    Pickle a python object into a serialized (pickle) object.
    @param filename: The name of the pickle object to generate/overwrite
    @param data: Python object to get pickled
    """

    try:
        with open(filename, 'wb+') as f:
            pickle.dump(data, f)
    except Exception as e:
        print(e)
        return 1
    return 0

def unit_test():
    import os
    import random
    data = {str(k): random.random() for k in range(100)}
    export_object("./database/test_database", data)
    assert import_object("./database/test_database") == data
    os.remove("./database/test_database")

if __name__ == "__main__":
    unit_test()
