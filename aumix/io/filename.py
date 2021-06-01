"""
filename.py

Filename retrival.

@author: Chan Wai Lou
"""

import re
import os


def id_filename_dict(folder_path, ids, regexes, sort_function):
    # Find all filenames
    filenames = [f for f in os.listdir(folder_path)]

    # Put the est file names in a dictionary and sort them in lists
    file_dict = dict()
    for i, _id in enumerate(ids):
        pattern = re.compile(regexes[i])
        matches = [f for f in filenames if pattern.match(f)]

        # Sort according to position
        matches.sort(key=sort_function)
        file_dict[_id] = matches

    return file_dict
