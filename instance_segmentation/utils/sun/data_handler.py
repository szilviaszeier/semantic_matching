import os
import re
import json
import glob
import logging

import plotly.express as px

subclass_to_class = {

    # All different types of walls
    "wall": ['wall',
             'brickwall', 'walls', 'whitewall', 'woodenwall', 'woodwall', 'rightwall', 'leftwall', 'walll',
             'whitebrickwall', 'windowwall', 'outsidewall', 'cornerwall', 'doorwall', 'partitionwalls', 'showerwall',
             'bluewall'
             ],

    # All different types of floors
    "floor": ['floor',
              'tilefloor', 'woodfloor', 'hardwoodfloor', 'flooring', 'woodenfloor', 'blackfloor', 'floor.',
              'warehousefloor', 'bathroomfloor', 'redfloor', 'hallwayfloor', 'woodenflooring', 'woodfloorboard',
              'displayfloor'
              ],

    # All different types of ceilings
    "ceiling": ['ceiling', 'ceilingwall', 'ceilings'],

    # chair, seat, & stool
    "chair": ['chair', 'chairs', 'seat',
              'officechair', 'deskchair', 'steelchair',
              'diningchair', 'lecturechairs', 'plasticchair', 'armchair', 'rollingchair', 'chairseat',
              'computerchair', 'cushionchair', 'wheelchair', 'revolvingchair', 'schoolchair', 'plasticchair',
              'conferencetablechair', 'high_chair', 'blackchair', 'chaircushion',
              'rockingchair', 'yellowchair', 'wheel_chair', 'whitechair',
              'stool', 'barstool'
              ],

    # table & desk
    "table": ['table', 'tables', 'desk', 'desktop',
              'coffeetable', 'studentdesk',
              'glasstable', 'foldingtable', 'dinningtable', 'tablewithdrawer', 'nighttable', 'endtable',
              'sidetable', 'bedsidetable', 'roundtable', 'computertable', 'diningtable', 'smalltable',
              'dressingtable', 'woodentable', 'worktable', 'desktable', 'banquettable', 'woodtable', 'teatable',
              'tablewithdrawers', 'desktables', 'rectangulartable', 'officetable', 'studytable', 'kitchentable',
              'steeltable', 'tvtable', 'tablewithshelf', 'table/desk', 'whitetable', 'atable', 'longtable',
              'desk/officetable', 'workingtable', 'desk/table', 'countertable', 'centertable', 'minitable',
              'changingtable', 'bed-sidetable', 'buffettable', 'readingtable',
              'workdesk', 'schooldesk'
              ],

    # All different types of doors
    "door": ['door', 'doors', 'glassdoor', 'woodendoor', 'adoor', 'doubledoors', 'slidingdoor', 'bathroomdoor',
             'storagedoor', 'doubledoor', 'doorglass', 'glassdoors', 'officedoor', 'wooddoor'],

    # All different types of windows
    "window": ['window',
               # 'windows', 'windowframe', 'glasswindow', 'woodenwindow', 'window_frame', 'windowglass',
               # 'glasswindows', 'officewindow', 'housewindows'
               ],

    # curtain & blinds
    "curtain": ['curtain',
                # 'curtains', 'shower_curtain', 'wallcurtain', 'windowcurtains',
                'blinds',
                # 'windowblinds', 'mimiblinds', 'miniblinds'
                ],

    # picture, poster, & painting
    "picture": ['picture', 'poster', 'pictures',
                # 'pictureframe', 'framedpicture', 'wallpicture',
                # 'pictureinframe', 'photosandpictures', 'pictureframes', 'pictureframe/art',
                'photo',
                # 'photoframe', 'photograph', 'photos', 'wallphoto',
                # 'wallphotos', 'photoframes',
                'painting', 'wallpainting',
                # 'paintings', 'posterboard',
                # 'wallposter',
                # 'posters'
                ],

    # All different types of mirrors
    "mirror": ['mirror', 'mirrorframe', 'wallmirror', 'bathroommirror', 'makeupmirror'],

    # All types of beds
    "bed": ['bed',
            # 'bedframe', 'bedsheet', 'bedsidetable', 'bunkbed', 'beds', 'babybed', 'toddlerbed'
            ],

    # All types of benches
    "bench": ['bench', 'woodenbench', 'piano_bench', 'benchseat', 'benchbase', 'leatherbench', 'benchseating',
              'benchbackrest', 'workbench', 'pianobench', 'boothbench', 'benches', 'laboratorybench', 'roundedbench',
              'longbench'],

    # sofa & couch
    "sofa": ['sofa',
             # 'sofachair', 'sofachairsingle', 'sofas', 'sofaset', 'roundsofa', 'sofa_chair', 'brownsofa',
             # 'cushionsofa', 'sectionalsofa', 'woodensofa', 'sofachairs', 'greensofa', 'leathersofa', 'cushionsofas',
             # 'smallsofa', 'bigsofa', 'sofaseat', 'whitesofa', 'blacksofa',
             'couch',
             # 'couchback', 'sectionalcouch', 'leathercouch', 'couchset', 'greycouch', 'blackcouch',
             # 'whitecouch', 'couches', 'sittingcouch', 'redcouch'
             ],

    # pillow & cushion
    "pillow": ['pillow',
               # 'pillows', 'throwpillow', 'throwpillows', 'decorativepillow', 'bedpillow',
               'cushion',
               # 'couchcushion',
               # 'seatcushion', 'sofacushion', 'cushions', 'chaircushion', 'couchpillow',
               # 'sofapillow',
               # 'roundpillow',
               # 'cylinderpillow', 'greenpillow'
               ],

    # monitor & tv
    "monitor": ['monitor', 'tv', 'computermonitor', 'screen', 'television', 'comptuermonitor', 'tvmonitor',
                # 'computerscreen', 'compuermonitors', 't.v',
                # 'projectorscreen', 'windowscreen', 'projectionscreen', 'projector_screen', 'whitescreen',
                # 'blindscreen', 'desktopscreen', 'blackscreen',
                # 'pcmonitor', 'flatscreentv', 'tvscreen', 'ledtv', 'tvmonitor', 'wall-mountedtv',
                # 'lcdtv'
                ],

    # pc & laptop
    "pc": ['desktopcomputer', 'pc', 'laptopcomputer', 'computer', 'laptop', 'desktop'],

    "sink": ['sink', 'bathroomsink', 'sinkbasin', ],

    "plant": ['plant',
              # 'plant_pot', 'pottedplant', 'plantpot', 'plants', 'planter', 'houseplant',
              # 'artificialplant', 'flowerplant', 'indoorplant', 'decorativeplant', 'fakeplant', 'pottedplants',
              # 'smallplant', 'largepottedplant'
              ],

    # cabinet, shelf, drawer, counter, dresser, nightstand, & closet
    "cabinet": ['cabinet',
                'filecabinet',
                'filingcabinet', 'cabinets', 'cabinetdrawer', 'cabinetdoor',
                'kitchencabinet', 'storagecabinet', 'metalcabinet', 'filingcabinets', 'cabinetry', 'cabinetdoors',
                'kitchencabinets', 'displaycabinet', 'organizercabinet', 'woodcabinet', 'uppercabinet', 'chinacabinet',
                'medicinecabinet', 'lowercabinet', 'wallcabinet', 'cabinetshelf', 'tablecabinet', 'curiocabinet',
                'storagecabinets', 'overheadcabinet', 'filecabinets', 'tvcabinet', 'lockedcabinet', 'mirroredcabinet',
                'overheadcabinets', 'uppercabinets', 'lowercabinets', 'sidecabinet', 'fillingcabinet',
                'cornerchinacabinet', 'woodencabinets', 'woodencabinet',

                'shelf', 'shelves', 'bookshelf',
                'woodshelf', 'shelfwithdoor', 'shelfsupport', 'woodenshelf',
                'wallshelf',
                'shoeshelf', 'lackwallshelf', 'metalshelf', 'displayshelf', 'hangingshelf', 'barshelf', 'shelfes',
                'deskshelf',
                'shelftop', 'woodshelfwithdoor', 'topwithshelfwithdoor', 'plasticshelf', 'booksshelf', 'shelfwithbooks',
                'bookshelves', 'bookshelves', 'bookstand', 'bookshelfcontainingbooks', 'shelfs', 'glassshelf',

                'drawer',
                'drawers', 'chestofdrawers', 'deskdrawer', 'drawerpull', 'dresserdrawer', 'shelfwithdrawer',
                'deskdrawers', 'kitchendrawer', 'storagedrawer', 'chesterdrawer', 'suspendeddrawer', 'opendrawer',
                'woodendrawer', 'tabledrawer', 'flatfiledrawers', 'filingcabinetdrawer', 'chesterdrawers',
                'deskdrawerknob',
                'officedrawer', 'drawerstand',

                # 'counter',
                # 'countertop', 'kitchencounter', 'counterbacking', 'counterbase', 'countertable',
                # 'sinkcounter',
                # 'bathroomcounter', 'labcounter', 'cabinetcounter', 'backcounter', 'woodcounter', 'tablecounter',
                # 'rearcounter',
                # 'undercountercabinet',

                'dresser',
                'dresserdrawer', 'dresserwithmirror', 'wooddresser', 'mirroreddresser', 'dressertop',
                'metaldresser', 'talldresser',

                # 'nightstand', 'night_stand',
                # 'closet', 'europeancloset', 'insidecloset'
                ],

    # trashcan, bin, & garbagecan
    "trashcan": ['trashcan', 'bin', 'bins', 'garbage_bin', 'garbagecan', 'garbage', 'recyclebin', 'dustbin',
                 'trashbin',
                 'trash', 'wastebin', 'garbage', 'garbagebin', 'recyclingbin', 'plasticgarbage',
                 'bluetrashcan', 'recycletrashcan', 'recycling_bin', 'plasticbin', 'recycle_bin', 'dusbin',
                 'recylingbin', 'wastepaperbin', 'bluerecyclebin', 'trashorrecycling', 'wastebasket',
                 'wastebox',
                 'waste', 'wastebascket', 'wastecan', 'wastepaperbin'
                 ],

    # carpet, rug, & mat
    "carpet": ['carpet', 'rug', 'floor_mat', 'mat', 'floormat', 'placemat', 'floormate', 'doormat', 'bathmat',
               'tablemat', 'yoga_mat', 'deskmat', 'floormats',
               'carpetedfloor', 'carpetfloor', 'redcarpet', 'bluecarpet', 'floorcarpet',
               'officecarpet', 'carpetfloors', 'carpeting', 'floor(carpet)', 'rolledcarpet',
               'arearug', 'rolleduprug', 'rugfloor', 'bathrug'],

    # toilet & urinal
    "toilet": ['toilet', 'toiletseat', 'westerntoilet', 'toiletbowl', 'urinal'],

    "bathtub": ['bathtub'],

    # kitchen appliance
    "microwave": ['microwave', 'microwaveoven', 'oldmicrowave'],

    "refrigerator": ['fridge', 'refridgerator', 'refrigerator', 'minifridge', 'refridge', 'freezer',
                     'refrigeratorfreezer', 'refrigeratordoor', 'cooler', 'watercooler',
                     ],

    "oven": ['oven',
             'toaster_oven', 'toasteroven', 'microoven', 'electricoven', 'ovenstove',
             'stove',
             'stoveburner', 'stovetop', 'gasstove', 'stovevent', 'stoveeye', 'gasstovetop', 'firegas',
             ],

    "object": [],  # all other objects

}

categories = ['wall', 'floor', 'ceiling', 'chair', 'table', 'bed', 'sofa', 'cabinet',
              'picture', 'pillow', 'monitor', 'window',
              'toilet', 'refrigerator']


def load_data(data_path, rgb_images, json_annotations):
    """ Load the training and validation sets (annotation, and image) paths.
    Args:
        data_path: String, indicating the relative or absolute main data directory.
        rgb_images: String, indicating the rgb images location extension in the main directory.
        json_annotations:String, indicating the annotation files location extension in the main directory.

    Returns:
        list of image paths, and a list of the corresponding annotation paths.

    """
    images_list = list(glob.iglob(data_path + rgb_images, recursive=True))
    annotations_list = list(glob.iglob(data_path + json_annotations, recursive=True))
    logging.info(f"Number of images: {len(images_list)}")
    logging.info(f"Number of json: {len(annotations_list)}")

    # sanity check
    for image, annotation in zip(images_list, annotations_list):
        if image.split("/")[-3] != annotation.split("/")[-3]:
            logging.warning(image)
            logging.warning(annotation)

    return images_list, annotations_list


def get_statistics(annotations_list):
    """ Retrieve unique objects with their distribution.
    Args:
        annotations_list: A list of annotation paths.

    Returns:
        a dictionary containing the unique objects with their count, and a list of erroneous files.
    """

    objects_count = {}
    error_files = []
    for file in annotations_list:
        try:
            data = json.load(open(file))
            objects_list = data["objects"]
            instance_list = data["frames"][0]["polygon"]
            objects = {index: list(obj.values())[0] for index, obj in enumerate(objects_list) if type(obj) == dict}

            for instance in instance_list:
                try:
                    obj_id = instance['object']
                    obj_name = objects[obj_id].lower().strip()

                    if obj_name not in objects_count.keys():
                        objects_count[obj_name] = 1
                    else:
                        objects_count[obj_name] += 1

                except KeyError as ke:
                    logging.exception(f"{ke}\n{file}")
                    error_files.append(file)

        except json.decoder.JSONDecodeError as je:
            logging.exception(f"{je}\n{file}")
            error_files.append(file)

    return objects_count, error_files


def plot_statistics(objects, top_k=50):
    """ Plot a data distribution as a bar chart

    Args:
        objects: A dictionary of order objects according to their occurrence (values).
        top_k: Maximum number of elements to plot.

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

        if type(instance['x']) == list:
            if len(instance['x']) > 2:
                regions.append({'x': instance['x'],
                                'y': instance['y'],
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

    def check_pattern(obj_name):
        """ Check if the object's name is with the required patterns
        1) object name ends with a "."
        2) object name starts with a "a"
        3) object name is followed by a number "1" "#1"
        4) object name starts with a color "whiteobject"
        5) object name starts with "firstobject" "secondobject"
        6) object names starts with relative location "rightobject" "leftobject"

        Returns:

        """

        # "." case #1
        new_key = get_key(obj_name[:-1])
        if new_key is not None:
            return new_key

        # "a" case #2
        new_key = get_key(obj_name[1:])
        if new_key is not None:
            return new_key

        # "ends with number" case #3
        m = re.search(r'\d+$', obj_name)
        if m is not None:
            new_key = get_key(obj_name.split(m.group())[0])
            if new_key is not None:
                return new_key

        # "starts with a color" case #4
        colors = ["black", "white", "red", "green", "blue", "yellow", "orange", "silver"]
        for color in colors:
            if obj_name.startswith(color):
                new_key = get_key(obj_name[len(color):])
                if new_key is not None:
                    return new_key

        # "starts with a number" case #5
        numbers = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
                   "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth",
                   "eighteenth", "nineteenth", "twentieth"]
        for number in numbers:
            if obj_name.startswith(number):
                new_key = get_key(obj_name[len(number):])
                if new_key is not None:
                    return new_key

        # "starts with a location" case #6
        locations = ["right", "left"]
        for location in locations:
            if obj_name.startswith(location):
                new_key = get_key(obj_name[len(location):])
                if new_key is not None:
                    return new_key

    files = {}
    new_objects = {key: 0 for key in subclass_to_class.keys()}
    for file, image_path in zip(annotations_list, images_list):
        try:

            data = json.load(open(file))
            objects_list = data["objects"]
            instance_list = data["frames"][0]["polygon"]
            objects = {index: list(obj.values())[0] for index, obj in enumerate(objects_list) if type(obj) == dict}
            regions = []

            for instance in instance_list:
                try:

                    obj_id = instance['object']
                    obj_name = objects[obj_id].lower().strip()
                    main_obj_name = get_key(obj_name)

                    if main_obj_name is not None:
                        add_region(instance, main_obj_name)
                    else:
                        main_obj_name = check_pattern(obj_name)

                        if main_obj_name is not None:
                            add_region(instance, main_obj_name)
                        else:
                            if "object" in categories:
                                add_region(instance, "object")

                except KeyError as ke:
                    logging.exception(f"{ke}\n{file}")

            if len(regions) > 0:
                filename = "/".join(image_path.split("/")[3:])
                files[file] = {"filename": filename, "regions": regions}
            else:
                logging.debug(f"No suitable objects were found in: {file}")

        except json.decoder.JSONDecodeError as je:
            logging.exception(f"{je}\n{file}")

        except UnicodeDecodeError as ue:
            logging.exception(f"{ue}\n{file}")

    return files, new_objects


def sort_dictionary(objects):
    """
    Args:
        objects: A dictionary to be sorted according to the values.

    Returns:
        sorted dictionary.

    """
    return dict(sorted(objects.items(), key=lambda item: -item[1]))


def save_annotations(annotations_list, images_list, data_path,
                     new_annotations_file="index.json", ):
    """ Save the new annotations in accordance with the newly constructed objects list.
    Args:
        annotations_list: List of the annotations paths
        images_list: List of the images paths.
        data_path: String, indication the location of the main data directory.
        new_annotations_file: The name of the file where the newly constructed annotations is to be saved.

    Returns:
        A dictionary containing the new objects struction with their distribution.
    """
    files, new_objects = subclass_to_class_mapping(images_list, annotations_list, subclass_to_class)
    with open(os.path.join(data_path, new_annotations_file), 'w') as outfile:
        json.dump(files, outfile, indent=4)

    logging.info(f"Number of processed JSON files: {len(files)}")
    return new_objects
