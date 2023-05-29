import os
import argparse
import datetime
import logging
import warnings
import torch
import glob

from utils.trainer import Train
from utils.trainer_utils import save_json

from detectron2.utils.logger import setup_logger
from detectron2.engine import launch

from shapely.errors import ShapelyDeprecationWarning

setup_logger()
FLAGS = []
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
random_seed = 123
import numpy as np

np.random.seed(random_seed)


def check_compatibility():
    torch_version = ".".join(torch.__version__.split(".")[:2])
    cuda_version = torch.__version__.split("+")[-1]
    logging.info(f"torch: {torch_version}; cuda: {cuda_version}")
    logging.info(f"{torch.cuda.device_count()}")


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-dp', '--data-path', type=str,
                        default="./datasets",
                        help='Main datasets directory')

    parser.add_argument('-dc', '--dictionaries-path', type=str,
                        default="./annotations/tmp",
                        help='JSON coco format annotations path')

    parser.add_argument('-ld', '--load-dictionaries', type=bool,
                        default=False,
                        help='Indicates whether to load saved data dictionaries or not')

    parser.add_argument('-aug', '--use-data-augmentation', type=bool,
                        default=False,
                        help='Indicates whether to augment the training data or not.')

    parser.add_argument('-prn', '--pre-trained-model', type=str,
                        default="Transfiner",
                        help='Pretrained model name')

    parser.add_argument('-pbb', '--pre-trained-backbone', type=str,
                        default="pre-trained",  # "R101-FPN-DCN"
                        help='Pretrained model backbone')

    parser.add_argument('-wn', '--workers-number', type=int,
                        default=torch.cuda.device_count(),
                        help='Number of workers')

    parser.add_argument('-ug', '--use-gpu', type=bool,
                        default=True,
                        help='Indicate whether to use gpu or cpu')

    parser.add_argument('-ipb', '--images-per-batch', type=int,
                        default=10,
                        help='Number of images per batch across all machines '
                             'This is also the number of training images per step')

    parser.add_argument('-lr', '--base-lr', type=int,
                        default=0.0005,
                        help='Learning rate')

    parser.add_argument('-it', '--max-iterations', type=int,
                        default=6000,
                        help='Number of iterations')

    parser.add_argument('-bs', '--batch-size', type=int,
                        default=512,
                        help='Number of regions per image used to train RPN')

    parser.add_argument('-out', '--output-dir', type=str,
                        default="./output",
                        help='Saved model output dir.')

    parser.add_argument('-th', '--score-threshold', type=float,
                        default=0.7,
                        help='Prediction score minimum threshold')

    parser.add_argument('-mn', '--model-name', type=str,
                        default="model_final.pth",
                        help='Saved model name')

    parser.add_argument('-aj', '--annotations-json', type=str,
                        default="1_improved_annotations_5.json",
                        help='Initial annotations file name')

    FLAGS, unparsed = parser.parse_known_args()

    return FLAGS


if __name__ == '__main__':
    check_compatibility()
    FLAGS = parse_arguments()

    categories = ['wall', 'floor', 'ceiling', 'chair', 'table',
                  'window', 'curtain', 'picture', 'bed', 'sofa',
                  'pillow', 'monitor', 'sink', 'trashcan', 'toilet',
                  'refrigerator', 'oven', 'bathtub', 'cabinet',
                  'object']

    dictionaries_names = ['train_dataset_dicts.json', 'val_dataset_dicts.json',
                          'test_dataset_dicts.json', 'dataset_dicts.json']

    pretrained_models_path = "./models"
    dataset_name = "indoor_objects"

    data_path = []
    dataset_dir = "./semantic_matching"
    used_method = ["supix_fast_slic_100_10"]

    environments_name = glob.glob(f"{dataset_dir}/*")
    for env_name in environments_name:
        env_states = glob.glob(f"{env_name}/*")
        for env_state in env_states:
            env_views = glob.glob(f"{env_state}/*")
            for segms in env_views:
                if segms.split("/")[-1] in used_method:
                    logging.info(f"Loading annotation file: {segms}")
                    data_path.append(segms)

    #
    model_name = FLAGS.pre_trained_model
    backbone = FLAGS.pre_trained_backbone
    dictionaries_path = FLAGS.dictionaries_path
    load_pre_existing_dicts = FLAGS.load_dictionaries
    output_dir = FLAGS.output_dir
    score_threshold = FLAGS.score_threshold
    loaders_number_of_workers = FLAGS.workers_number
    use_gpu = FLAGS.use_gpu
    image_per_batch = FLAGS.images_per_batch
    base_lr = FLAGS.base_lr
    max_iterations = FLAGS.max_iterations
    batch_size_per_image = FLAGS.batch_size
    use_data_augmentation = FLAGS.use_data_augmentation
    annotations_json = FLAGS.annotations_json

    model_ext = datetime.datetime.now().strftime("%m-%d-%Y, %H:%M:%S")
    output_dir = f"{output_dir}/{model_name}/{dictionaries_path.split('/')[-1]}/{backbone}/{used_method[0]}/a"

    trainer = Train(pretrained_models_path, model_name, backbone,
                    categories, data_path, dictionaries_path, dictionaries_names,
                    dataset_name,
                    load_pre_existing_dicts, output_dir, score_threshold,
                    loaders_number_of_workers, use_gpu, image_per_batch,
                    base_lr, max_iterations, batch_size_per_image, use_data_augmentation, annotations_json,
                    None)

    try:
        launch(
            trainer.train(),
            torch.cuda.device_count(),
            num_machines=1,
            machine_rank=0,
            args=(None,),
        )
    except Exception as ex:
        logging.exception(ex)

    validation_results = trainer.validate()
    save_json(validation_results, file_path=f"{output_dir}/test_results.json")
