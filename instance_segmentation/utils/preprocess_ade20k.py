import argparse
import logging

from utils.logging_settings import setup_loging_config
from utils.ade.data_handler import save_annotations, get_statistics, sort_dictionary, plot_statistics

setup_loging_config()


def main():
    parser = argparse.ArgumentParser(description="process ADE20K dataset.")

    parser.add_argument('-dp', '--data-path', type=str, default="./datasets/ADE20K_2021_17_01/images/ADE/",
                        help='Training and validation sets path')

    parser.add_argument('-ja', '--json-annotations', type=str, default='**/*.json',
                        help='JSON annotations extension')

    parser.add_argument('-in', '--index-name', type=str, default='index_12.json',
                        help='Processed JSON annotation file name')

    FLAGS, unparsed = parser.parse_known_args()

    data_path = FLAGS.data_path
    json_annotations = FLAGS.json_annotations
    index_file = FLAGS.index_name

    category_dirs = ["home_or_hotel", "work_place", "shopping_and_dining", "transportation", "cultural"]

    objects = get_statistics(data_path, json_annotations, category_dirs)
    new_objects = save_annotations(json_annotations, category_dirs, data_path, index_file)

    plot_statistics(sort_dictionary(objects), top_k=50)
    plot_statistics(sort_dictionary(new_objects))

    logging.info(f"Number of 'unique' objects: {len(objects.keys())}")
    logging.info(f"Number of all instance: {sum(objects.values())}")

    logging.info(f"Number of selected objects: {len(new_objects.keys())}")
    logging.info(f"Number of selected instance: {sum(new_objects.values())}")


if __name__ == '__main__':
    main()
