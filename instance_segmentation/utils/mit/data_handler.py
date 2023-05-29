import os
import re
import json
import glob
import logging

import plotly.express as px

subclass_to_class = {

    # All different types of walls
    "wall": ['wall'],

    # All different types of floors
    "floor": ['floor'],

    # All different types of ceilings
    "ceiling": ['ceiling'],

    # chair, seat, & stool
    "chair": ['chair'],

    # table & desk
    "table": ['table', 'desk'],

    # All different types of doors
    "door": ['door'],

    # All different types of windows
    "window": ['window'],

    # curtain & blinds
    "curtain": ['curtain'],

    # picture, poster, & painting
    "picture": ['picture', 'poster', 'photo', 'painting'],

    # All different types of mirrors
    "mirror": ['mirror'],

    # All types of beds
    "bed": ['bed'],

    # All types of benches
    "bench": ['bench'],

    # sofa & couch
    "sofa": ['sofa', 'couch'],

    # pillow & cushion
    "pillow": ['pillow', 'cushion'],

    # monitor & tv
    "monitor": ['monitor', 'tv', 'screen', 'television'],

    # pc & laptop
    "pc": ['desktop', 'computer', 'laptop', 'desktop'],

    "sink": ['sink'],

    "plant": ['plant', 'plants'],

    # cabinet, shelf, drawer, counter, dresser, nightstand, & closet
    "cabinet": ['cabinet',
                'shelf',
                'drawer',
                'counter',
                'dresser',
                ],

    # trashcan, bin, & garbagecan
    "trashcan": ['bin',
                 'trash',
                 ],

    # carpet, rug, & mat
    "carpet": ['carpet', 'rug'],

    # toilet & urinal
    "toilet": ['toilet'],

    "bathtub": ['bathtub'],

    # kitchen appliance
    "microwave": ['microwave'],

    "refrigerator": ['fridge', 'refridgerator', 'refrigerator'],

    "oven": ['oven'],

    "object": [],  # all other objects

}

categories = ['wall', 'floor', 'ceiling', 'chair', 'table', 'bed', 'sofa', 'cabinet',
              'picture', 'pillow', 'monitor', 'window',
              'toilet', 'refrigerator']


def load_data(data_path, rgb_images, json_annotations):
    """Load the training and validation sets.

    Args:
        data_path (str): The main data directory path.
        rgb_images (str): The extension for RGB image files.
        json_annotations (str): The extension for JSON annotation files.

    Returns:
        tuple: A tuple containing lists of image paths and annotation paths.

    """

    annotations_list = list(glob.iglob(data_path + json_annotations, recursive=True))
    images_list = []
    for annotation in annotations_list:
        folders = annotation.split("/")
        tmp = "/".join(folders[:3] + ["indoorCVPR_09", "Images"] + folders[5:6])
        tmp += "/" + folders[-1].split(".")[0] + ".jpg"
        images_list.append(tmp)
    logging.info(f"Number of images: {len(images_list)}")
    logging.info(f"Number of json: {len(annotations_list)}")

    return images_list, annotations_list


def get_statistics(annotations_list):
    """Retrieve unique objects with their distribution.

    Args:
        annotations_list (list): A list of annotation paths.

    Returns:
        tuple: A tuple containing a dictionary of unique objects and their counts, and a list of erroneous files.

    """

    objects_count = {}
    error_files = []
    for file in annotations_list:
        try:
            data = json.load(open(file))
            objects_list = data['annotation']["object"]
            # print(file)

            for instance in objects_list:
                try:
                    # obj_id = instance['id']
                    obj_name = str(instance['name']).strip().lower()

                    if obj_name not in objects_count.keys():
                        objects_count[obj_name] = 1
                    else:
                        objects_count[obj_name] += 1

                except KeyError as ke:
                    logging.exception(f"{ke}\n{file}")
                    error_files.append(file)
                except TypeError as te:
                    logging.exception(f"{te}\n{file}")
                    error_files.append(file)
                except AttributeError as ae:
                    logging.exception(f"{ae}\n{file}")
                    error_files.append(file)
                    print(file)

        except json.decoder.JSONDecodeError as je:
            logging.exception(f"{je}\n{file}")
            error_files.append(file)
        except KeyError as ke:
            logging.exception(f"{ke}\n{file}")
            error_files.append(file)

    return objects_count, error_files


def plot_statistics(objects, top_k=50):
    """Plot a bar chart to visualize object distribution.

    Args:
        objects_count (dict): A dictionary of unique objects and their counts.
        top_k (int, optional): The maximum number of elements to plot. Defaults to None.

    """
    objects = dict(list(objects.items())[:top_k])
    fig = px.bar(x=objects.keys(), y=objects.values(), color=objects.values(), text_auto='.2s',
                 title="Data distribution")
    fig.show()


def subclass_to_class_mapping(images_list, annotations_list, subclass_to_class):
    def add_region(instance, obj_name):
        """ Add a new object to the list of required regions

        Args:
            instance: A dictionary contain the object name, with the corresponding Xs, and Ys.
            obj_name: The name of the new category.

        """

        if type(instance) == list:
            if len(instance) > 2:
                regions.append({'x': [int(inst['x']) for inst in instance],
                                'y': [int(inst['y']) for inst in instance],
                                "object_name": obj_name})
            new_objects[obj_name] += 1

    def get_key(name):
        """ Check of the selected object dictionary and return the main object name (if any)
        Args:
            name: The name of the object to be checked.

        Returns:
            The main object name (if any).
        """

        for key, value in subclass_to_class.items():
            if key in categories:
                if name == key or name in value:
                    return key
        return None

    files = {}
    new_objects = {key: 0 for key in subclass_to_class.keys()}
    for file, image_path in zip(annotations_list, images_list):
        try:

            data = json.load(open(file))
            objects_list = data['annotation']["object"]
            filename = "/".join(image_path.split("/")[3:])
            regions = []

            for instance in objects_list:

                try:
                    obj_name = str(instance['name']).strip().lower()
                    main_obj_name = get_key(obj_name)
                    if main_obj_name is not None:
                        add_region(instance['polygon']['pt'], main_obj_name)

                    else:
                        if "object" in categories:
                            add_region(instance['polygon']['pt'], "object")

                except KeyError as ke:
                    logging.exception(f"{ke}\n{file}")
                except TypeError as te:
                    logging.exception(f"{te}\n{file}")
                except AttributeError as ae:
                    logging.exception(f"{ae}\n{file}")

            if len(regions) > 0:
                files[file] = {"filename": filename, "regions": regions}

            else:
                logging.debug(f"No suitable objects were found in: {file}")

        except json.decoder.JSONDecodeError as je:
            logging.exception(f"{je}\n{file}")

        except UnicodeDecodeError as ue:
            logging.exception(f"{ue}\n{file}")

        except KeyError as ke:
            logging.exception(f"{ke}\n{file}")

    return files, new_objects


def sort_dictionary(objects):
    """Sort a dictionary by values in descending order.

    Args:
        dictionary (dict): The dictionary to be sorted.

    Returns:
        dict: The sorted dictionary.

    """
    return dict(sorted(objects.items(), key=lambda item: -item[1]))


def save_annotations(annotations_list, images_list, data_path,
                     new_annotations_file="index.json", ):
    """Generate new annotations based on the mapping and save them to a JSON file.

    Args:
        images_list (list): A list of image paths.
        annotations_list (list): A list of annotation paths.
        data_path (str): The main data directory path.
        new_annotations_file (str, optional): The name of the new annotations file. Defaults to None.

    Returns:
        dict: A dictionary of the new objects structure with their distribution.

    """
    files, new_objects = subclass_to_class_mapping(images_list, annotations_list, subclass_to_class)
    with open(os.path.join(data_path, new_annotations_file), 'w') as outfile:
        json.dump(files, outfile, indent=4)

    logging.info(f"Number of processed JSON files: {len(files)}")
    return new_objects
