import json
import logging
import os

import numpy as np

categories = ['wall', 'floor', 'ceiling', 'chair', 'table',
              'window', 'curtain', 'picture', 'bed', 'sofa',
              'pillow', 'monitor', 'sink', 'trashcan', 'toilet',
              'refrigerator', 'oven', 'bathtub', 'cabinet',
              'object']

category_colors = [[119, 119, 119], [131, 243, 244], [190, 190, 255], [113, 114, 54], [255, 150, 255],
                   [33, 163, 0], [0, 255, 150], [10, 180, 255], [255, 255, 150], [176, 0, 0],
                   [255, 209, 24], [55, 163, 152], [115, 72, 70], [34, 64, 87], [234, 195, 193],
                   [212, 79, 192], [115, 72, 70], [131, 57, 52], [157, 28, 137],
                   [0, 69, 255]]


def setup_logging_config(filename=None, level=logging.INFO,
                         format='%(asctime)s %(name)s - %(levelname)s - %(message)s',
                         datefmt='%d-%b-%y %H:%M:%S'):
    """Sets up the logging configuration.

    Args:
        filename: The name of the log file.
        level: The logging level.
        format: The log message format.
        datefmt: The log date format.
    """
    logging.basicConfig(
        filename=filename,
        filemode='w',
        level=level,
        format=format,
        datefmt=datefmt
    )


def load_annotations(dataset_dir, annotations_file_name):
    """Loads the annotations from a JSON file.

    Args:
        dataset_dir: The directory path containing the dataset.
        annotations_file_name: The name of the JSON file containing the annotations.

    Returns:
        The loaded annotations as a dictionary.
    """
    return load_json(os.path.join(dataset_dir, annotations_file_name))


def save_json(data, file_path):
    """Saves data as a JSON file.

    Args:
        data: The data to be saved.
        file_path: The file path to save the JSON file.
    """
    with open(file_path, 'w') as outfile:
        json.dump(data, outfile, indent=4, cls=NpEncoder)


def load_json(file_path):
    """Loads data from a JSON file.

    Args:
        file_path: The file path to load the JSON file from.

    Returns:
        The loaded data as a dictionary.
    """
    with open(file_path) as json_file:
        data = json.load(json_file)
    return data


class NpEncoder(json.JSONEncoder):
    """Helper class to handle JSON encoding of numpy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def color_to_index(color_code):
    """Converts a color code to its corresponding index.

    Args:
        color_code: The color code to convert.

    Returns:
        The index of the color code in the category colors list.
    """
    return category_colors.index(color_code)
