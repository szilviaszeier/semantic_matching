import logging
import os
import random

import cv2
import numpy as np
from detectron2.utils.visualizer import Visualizer
from imantics import Mask

from .helper_functions import color_to_index, save_json, setup_logging_config

setup_logging_config()


def calculate_polygon_area(x, y):
    """Calculates the area of a polygon using its vertices.

    Args:
        x: List of x-coordinates of the polygon vertices.
        y: List of y-coordinates of the polygon vertices.

    Returns:
        The area of the polygon.
    """
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def annotate_dataset(gt_dir, image_dir, annotations_file_name, sample=5):
    """Annotates the dataset with ground truth information.

    Args:
        gt_dir: Directory path containing ground truth images.
        image_dir: Directory path containing corresponding RGB images.
        annotations_file_name: Name of the output annotations file.
        sample: Sampling interval for processing the images.

    Returns:
        A list of dictionaries containing the annotations for each image.
    """
    annotations = []
    counter = -1
    sample_counter = 0
    occupied_threshold = 1 / 100

    for id, gt_filename in enumerate(os.listdir(gt_dir)):
        try:
            if not gt_filename.endswith(".png"):
                continue

            counter += 1
            if counter % sample != 0:
                continue

            try:
                gt = cv2.imread(os.path.join(gt_dir, gt_filename), 1)
                image_height, image_width, _ = gt.shape
            except AttributeError as ae:
                logging.exception(
                    f"An exception has occurred while reading the image: {os.path.join(gt_dir, gt_filename)}\n{ae}")
                continue

            sample_counter += 1
            image_info = {
                "file_name": os.path.join(image_dir, gt_filename),
                "image_id": id,
                "height": image_height,
                "width": image_width,
                "annotations": [],
            }
            arra = gt[:, :, 1]
            color_codes = np.unique(gt.reshape(-1, gt.shape[2]), axis=0).tolist()

            for index, color_code in enumerate(color_codes):
                if color_code == [0, 0, 0]:
                    continue

                ground_truth_binary_mask = np.where(arra != color_code[1], 0, 1).astype(np.uint8)
                occupied_pixels = np.sum(ground_truth_binary_mask)
                occupied_percentage = occupied_pixels / (
                        ground_truth_binary_mask.shape[0] * ground_truth_binary_mask.shape[1])

                if occupied_percentage < occupied_threshold:
                    continue

                polygons = Mask(ground_truth_binary_mask).polygons()

                broken = False
                for tmp in polygons.segmentation:
                    if len(tmp) < 5:
                        broken = True
                        break
                if broken:
                    continue

                annotation = {
                    "segmentation": polygons.segmentation,
                    "iscrowd": 0,
                    "bbox": np.array(list(polygons.bbox())),
                    "bbox_mode": 0,
                    "category_id": color_to_index(color_code),
                }

                image_info["annotations"].append(annotation)
            annotations.append(image_info)

        except ValueError as VE:
            logging.exception(VE)
            continue

    save_json(annotations, os.path.join(gt_dir, f"{annotations_file_name}"))
    logging.info(f"Number of samples: {sample_counter}")

    return annotations


def display_image_samples(dataset_dicts, number_of_examples=20):
    """Displays a random selection of image samples with annotations.

    Args:
        dataset_dicts: A list of dictionaries containing the information needed to display the data.
        number_of_examples: Number of examples to display.
    """
    for d in random.sample(dataset_dicts, number_of_examples):
        file_path = d["file_name"]
        img = cv2.imread(file_path)

        visualizer = Visualizer(img[:, :, ::-1], metadata=None, scale=1.5)
        out = visualizer.draw_dataset_dict(d)
        cv2.imshow("Segmentation Example", out.get_image()[:, :, ::-1])
        cv2.waitKey()


def display_instance_samples(dataset_dir, dataset_dicts, number_of_examples=10):
    """Displays a random selection of instance samples with annotations.

    Args:
        dataset_dir: Directory path containing the dataset images.
        dataset_dicts: A list of dictionaries containing the information needed to display the data.
        number_of_examples: Number of examples to display.
    """
    for d in random.sample(dataset_dicts, number_of_examples):
        file_path = os.path.join(dataset_dir + "/rgb", f'{d["file_name"].split(".")[0]}.png')
        d_copy = d.copy()

        for ann in d['annotations']:
            d_copy['annotations'] = [ann]

            img = cv2.imread(file_path)
            visualizer = Visualizer(img[:, :, ::-1], metadata=None, scale=0.5)
            out = visualizer.draw_dataset_dict(d_copy)
            cv2.imshow("Segmentation Example", out.get_image()[:, :, ::-1])
            cv2.waitKey()
