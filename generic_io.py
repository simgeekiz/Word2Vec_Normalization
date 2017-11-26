#!/usr/bin/env python3

"""Defines the functions to load and save data from various sources.





"""

import os
import json
import pickle
import logging
from datetime import datetime
import json_lines

LOGGER = logging.getLogger('generic_io')

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='mail.log',
                    filemode='w')

def load_from_file(file_path, file_format='auto', broken=False):
    """Loads data from a file and returns the data.

    Supported file formats; JSON, JSON-lines and Pickle.

    Parameters
    ----------
    file_path : string
        Give the path of the file to be loaded.

    file_format : string, optional (default='auto')
        The format of the file to be loaded. If 'auto' has given, it decides
        on the file format by looking at the file extension.

    broken : bool, optional (default=False)
        Give True, if the JSON-line file is broken for some reason (e.g. stoping a writing
        process in the half). For more information please see
        https://github.com/TeamHG-Memex/json-lines/blob/master/README.rst#usage

    Returns
    -------
    data_ : list, JSON, or object
        The object that contains the data loaded from the file. If the file
        format is JSON-lines, returned object is a list.
    """

    data_ = []
    file_format = file_format.lower()

    if file_format == 'auto':
        _, file_extension = os.path.splitext(file_path)
        file_format = file_extension[1:]

    if file_format in ['json']:
        try:
            with open(file_path, 'r') as json_file:
                data_ = json.load(json_file)
        except FileNotFoundError as exception:
            LOGGER.error('Could not load the data from the file. ' + str(exception))

    elif file_format in ['jsonlines', 'json-lines', 'jsonl', 'jl', 'jsons']:
        try:
            with json_lines.open(file_path, broken=broken) as json_lines_file:
                for item in json_lines_file:
                    data_.append(item)
        except FileNotFoundError as exception:
            LOGGER.error('Could not load the data from the file. ' + str(exception))

    elif file_format in ['pickle', 'pkl', 'pcl']:
        try:
            with open(file_path, 'rb') as pickle_file:
                data_ = pickle.load(pickle_file)
        except FileNotFoundError as exception:
            LOGGER.error('Could not load the data from the file. ' + str(exception))

    else:
        LOGGER.error('File format is not supported: ' + file_format)
        raise IOError('File format is not supported: ' + file_format)

    return data_

def save_to_file(data_, file_path, file_format='auto', include_timestamp=False):
    """Saves data to a file.

    Supported file formats; JSON and Pickle.

    Parameters
    ----------
    data_ : object
        The data that should be written to the file.

    file_path : string
        Give the path of the folder that the file should be saved under.

    file_format : string, optional (default='auto')
        Define the format you want to save the file as. If 'auto' has given, it
        looks for an extension in the filename.

    include_timestamp : bool, (default=False)
        Give True if you want to add a timestamp to the file name
    """

    file_format = file_format.lower()

    folder_path = os.path.dirname(file_path)
    os.makedirs(folder_path, exist_ok=True)

    file_name = os.path.basename(file_path)
    file_name_wo_ext, file_extension = os.path.splitext(file_name)

    if file_format == 'auto':
        file_format = file_extension[1:]

    if include_timestamp:
        now = datetime.utcnow().replace(second=0, microsecond=0).isoformat()
        file_name_wo_ext = file_name_wo_ext + '_' + now

    file_name = file_name_wo_ext + '.' + file_format
    file_path = os.path.join(folder_path, file_name)

    if file_format in ['json']:
        with open(file_path, 'w') as json_file:
            json.dump(data_, json_file)

    elif file_format in ['pickle', 'pkl', 'pcl']:
        with open(file_path, 'wb') as pickle_file:
            pickle.dump(data_, pickle_file)

    else:
        LOGGER.error("File format is not supported: " + file_format)
        raise IOError("File format is not supported: " + file_format)

    LOGGER.info("The data saved in a file named '" + file_name + "' in '"
                + file_format + "' file format, under '" + folder_path + "'")
