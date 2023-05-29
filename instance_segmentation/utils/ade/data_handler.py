import os
import glob
import json
import logging

import plotly.express as px

subclass_to_class = {
    "wall": ['wall'],

    "floor": ['floor'],

    "ceiling": ['ceiling'],

    "chair": ['chair',
              'armchair', 'swivel chair', 'chairs',
              'rocking chair', 'dental chair', 'deck chair',
              'wheelchair', 'folding chair', 'baby chair', 'reclining chair', 'folding chairs',
              'miniature chair', 'dental swivel chair', 'desk chair', 'chairr', 'steel chair', 'small chair',
              'chairs lift',
              'seat',
              'seat base', 'seats', 'seating', 'swinging seat',
              'stool',
              'stools'
              ],

    "table": ['table',
              'night table', 'side table', 'coffee table', 'console table', 'work table', 'small table',
              'tables', 'dressing table',
              'desk',
              'desks', 'writing desk'
              ],

    "door": ['door',
             'double door', 'sliding door', 'doors', 'folding door', 'shower door', 'folding doors'
             ],

    "window": ['window',
               'windows', 'shop window', 'office window'
               ],

    "curtain": ['curtain',
                'curtains',
                'cover curtain', 'shower curtain', 'curtain cover', 'curtain support'
                                                                    'blind',
                'blinds',
                'blinder'
                ],

    "picture": ["picture", "pictures", ],

    "mirror": ['mirror', 'mirrors'],

    "bed": ['bed',
            'sofa bed', 'beds'
            ],

    "bench": ['bench', 'workbench', 'lab bench', 'benches'],

    "sofa": ['sofa', 'sofa bed'],

    "pillow": ['pillow', 'back pillow',
               'pillows',
               'cushion', 'seat cushion',
               'cushions', 'back cushion'
               ],

    "monitor": ['monitor',
                'monitors', 'television',
                'tele vision'
                'screen',
                'projection screen'
                ],

    "pc": ['computer', 'computer case', 'laptop'],

    "sink": ['sink', 'dental sink'],

    "plant": ['plant',
              'plant pot', 'plants', 'plant pots', 'planter', 'pendant plant'
              ],

    "cabinet": ['cabinet',
                'cabinets',
                'filing cabinet', 'file cabinet',
                'control cabinet', 'file cabinets',
                'shelf',
                'shower shelf', 'shelves',
                'drawer',
                'chest of drawers', 'trolley drawer',
                'countertop', 'countertop', 'counters',
                'dresser', 'closet'
                ],

    "trashcan": ['trash can'],

    "carpet": ['carpet', 'carpets'],

    "toilet": ['toilet'],

    "bathtub": ['bathtub'],

    "microwave": ['microwave'],

    "refrigerator": ['refrigerator', 'water cooler', 'cooler'],

    "oven": ['oven', 'toaster oven', 'stove'],

    "object": [],
}

categories = ['wall', 'floor', 'ceiling', 'chair', 'table', 'bed', 'sofa', 'cabinet',
              'picture', 'pillow', 'monitor', 'window',
              'toilet', 'refrigerator']


def load_data(data_path, json_annotations):
    """Load the training and validation sets (annotation and image) paths.

    Args:
        data_path (str): String indicating the relative or absolute main data directory.
        json_annotations (str): String indicating the annotation files location extension in the main directory.

    Returns:
        list: List of image paths.
        list: List of corresponding annotation paths.
    """
    annotations_list = list(glob.iglob(data_path + json_annotations, recursive=True))
    logging.info(f"Number of json: {len(annotations_list)}")

    return annotations_list


def plot_statistics(objects, top_k=50):
    """Plot a data distribution as a bar chart.

    Args:
        objects (dict): Dictionary of ordered objects according to their occurrence (values).
        top_k (int): Maximum number of elements to plot.
    """
    objects = dict(list(objects.items())[:top_k])
    fig = px.bar(x=objects.keys(), y=objects.values(), color=objects.values(), text_auto='.2s',
                 title="Data distribution")
    fig.show()


def get_statistics(data_path, json_annotations, category_dirs):
    """Retrieve unique objects with their distribution.

    Args:
        data_path (str): String indicating the relative or absolute main data directory.
        json_annotations (list): List of annotation paths.
        category_dirs (list): List containing the names of required environments (home and offices).

    Returns:
        dict: Dictionary containing the unique objects with their count.
        list: List of erroneous files.
    """
    annotations_list = load_data(data_path, json_annotations)
    objects_count = {}
    for annotation in annotations_list[:]:
        if annotation.split("/")[-3] in category_dirs:
            try:
                json_file = json.load(open(annotation))['annotation']
                objects = json_file['object']
                for obj in objects:
                    obj_name = obj['raw_name'].lower().strip()
                    if obj_name not in objects_count.keys():
                        objects_count[obj_name] = 1
                    else:
                        objects_count[obj_name] += 1
            except UnicodeDecodeError as ue:
                logging.info(annotation)
                logging.info(ue)
                continue
    logging.info(f"Number of 'unique' objects: {len(objects_count.keys())}")

    return objects_count


def subclass_to_class_mapping(data_path, json_annotations, category_dirs):
    """Map subclass names to their corresponding class names in the annotation files.

    Args:
        data_path (str): String indicating the relative or absolute main data directory.
        json_annotations (list): List of annotation paths.
        category_dirs (list): List containing the names of required environments (home and offices).

    Returns:
        dict: Dictionary containing the mapped objects structure with their distribution.
        dict: Dictionary containing the counts of newly constructed objects.
    """

    def add_region(instance, obj_name):
        """Add a new object to the list of required regions.

        Args:
            instance (dict): Dictionary containing the object name, with the corresponding Xs and Ys.
            obj_name (str): The name of the new category.
        """
        if isinstance(instance['x'], list):
            if len(instance['x']) > 2:
                regions.append({'x': instance['x'], 'y': instance['y'], "object_name": obj_name})
            new_objects[obj_name] += 1

    def get_key(name):
        """Check the selected object dictionary and return the main object name (if any).

        Args:
            name (str): The name of the object to be checked.

        Returns:
            str: The main object name (if any).
        """
        for key, value in subclass_to_class.items():
            if key in categories:
                if name == key or name in value:
                    return key
        return None

    annotations_list = load_data(data_path, json_annotations)
    files = {}
    new_objects = {key: 0 for key in subclass_to_class.keys()}

    for annotation in annotations_list:
        if annotation.split("/")[-3] in category_dirs:
            try:
                json_file = json.load(open(annotation))['annotation']
                instances = json_file['object']
                regions = []

                for instance in instances:
                    obj_name = instance['raw_name'].lower().strip()
                    main_obj_name = get_key(obj_name)
                    if main_obj_name is not None:
                        add_region(instance['polygon'], main_obj_name)
                    else:
                        if "object" in categories:
                            add_region(instance['polygon'], "object")

                    if len(regions) > 0:
                        filename = ("/".join(annotation.split("/")[5:]))[:-4]
                        filename += "jpg"
                        files[annotation] = {"filename": filename, "regions": regions}
            except UnicodeDecodeError as ue:
                continue

    return files, new_objects


def sort_dictionary(objects):
    """Sort a dictionary according to the values.

    Args:
        objects (dict): Dictionary to be sorted.

    Returns:
        dict: Sorted dictionary.
    """
    return dict(sorted(objects.items(), key=lambda item: -item[1]))


def save_annotations(json_annotations, category_dirs, data_path, new_annotations_file="index.json"):
    """Save the new annotations in accordance with the newly constructed objects list.

    Args:
        json_annotations (list): List of the annotations paths.
        category_dirs (list): List of the images paths.
        data_path (str): String indicating the location of the main data directory.
        new_annotations_file (str): The name of the file where the newly constructed annotations will be saved.

    Returns:
        dict: Dictionary containing the new objects structure with their distribution.
    """
    files, new_objects = subclass_to_class_mapping(data_path, json_annotations, category_dirs)
    with open(os.path.join(data_path, new_annotations_file), 'w') as outfile:
        json.dump(files, outfile, indent=4)

    logging.info(f"Number of processed JSON files: {len(files)}")
    return new_objects
