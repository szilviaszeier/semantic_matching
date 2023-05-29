import json
import numpy as np


class NpEncoder(json.JSONEncoder):
    """
    Helper class to handle JSON encoding of NumPy arrays and data types.
    """

    def default(self, obj):
        """
        Override the default method to handle NumPy types during JSON encoding.

        Args:
            obj: The object to encode.

        Returns:
            JSON-compatible representation of the object.
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def save_json(data, file_path):
    """
    Save data dictionaries to a JSON file.

    Args:
        data (dict): A dictionary containing the dataset annotations.
        file_path (str): The location and name of the JSON file to be saved.
    """
    with open(file_path, 'w') as outfile:
        json.dump(data, outfile, indent=4, cls=NpEncoder)


def load_json(file_path):
    """
    Load data dictionaries from a JSON file.

    Args:
        file_path (str): The location and name of the JSON file to be loaded.

    Returns:
        data (dict): The loaded data as a dictionary.
    """
    with open(file_path) as json_file:
        data = json.load(json_file)
    return data
