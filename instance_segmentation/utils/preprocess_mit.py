import argparse
import logging

from utils.logging_settings import setup_loging_config
from utils.mit.data_handler import load_data, plot_statistics, get_statistics, save_annotations, sort_dictionary

setup_loging_config()


def main():
    parser = argparse.ArgumentParser(description="process SUNRGBD dataset.")

    parser.add_argument('-dp', '--data-path', type=str,
                        default="./datasets/MIT_Indoor_Scenes",
                        help='Training and validation sets paths')

    parser.add_argument('-ja', '--json-annotations', type=str,
                        default='/indoorCVPR_09annotations/Annotations/**/*.json',
                        help='JSON annotations extension')

    parser.add_argument('-ri', '--rgb-images', type=str,
                        default='indoorCVPR_09/Images/**/*.jpg',
                        help='RGB images extension')

    parser.add_argument('-in', '--index-name', type=str,
                        default='index_12.json',
                        help='Processed JSON annotation file name')

    FLAGS, unparsed = parser.parse_known_args()

    data_path = FLAGS.data_path
    rgb_images = FLAGS.rgb_images
    json_annotations = FLAGS.json_annotations
    index_file = FLAGS.index_name

    images_list, annotations_list = load_data(data_path, rgb_images, json_annotations)
    objects, error_files = get_statistics(annotations_list[:])
    objects = sort_dictionary(objects)

    new_objects = sort_dictionary(save_annotations(annotations_list, images_list, data_path, index_file))

    logging.info(f"Number of 'unique' objects: {len(objects.keys())}")
    logging.info(f"Number of all instance: {sum(objects.values())}")

    logging.info(f"Number of selected objects: {len(new_objects.keys())}")
    logging.info(f"Number of selected instance: {sum(new_objects.values())}")

    plot_statistics(objects)
    plot_statistics(new_objects)


if __name__ == '__main__':
    main()