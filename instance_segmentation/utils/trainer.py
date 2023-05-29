import json
import logging
import os
import random
import time

import cv2
import torch
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog
from detectron2.data import build_detection_test_loader
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    default_setup,
    default_writers,
    hooks,
)
from detectron2.engine import DefaultPredictor
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import (
    COCOEvaluator,
    verify_results,
)
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.modeling import build_model
from detectron2.structures import BoxMode
from detectron2.utils import comm
from detectron2.utils.visualizer import Visualizer
from utils.logging_settings import setup_loging_config
from utils.trainer_utils import CustomTrainer, DefaultTrainer, save_json, load_json, check_files, \
    ValidationLoss, CustomAugTrainer, AsyncPredictor

setup_loging_config()

logger = logging.getLogger("detectron2")

random_seed = 123
import numpy as np

np.random.seed(random_seed)


class Train:
    def __init__(self, pretrained_models_path, model_name, backbone,
                 categories, data_path, dictionaries_path, dictionaries_names,
                 dataset_name,
                 load_pre_existing_dicts, output_dir, score_threshold,
                 loaders_number_of_workers, use_gpu, image_per_batch,
                 base_lr, max_iterations, batch_size_per_image, use_data_augmentation, annotations_json, args):

        self.pretrained_models_path = pretrained_models_path
        self.model_name = model_name
        self.backbone = backbone
        self.categories = categories
        self.data_path = data_path
        self.dictionaries_path = dictionaries_path
        self.dictionaries_names = dictionaries_names
        self.dataset_name = dataset_name
        self.train_set_name = f"{dataset_name}_train"
        self.validation_set_name = f"{dataset_name}_val"
        self.test_set_name = f"{dataset_name}_test"
        self.load_pre_existing_dicts = load_pre_existing_dicts
        self.loaders_number_of_workers = loaders_number_of_workers
        self.use_gpu = use_gpu
        self.image_pre_batch = image_per_batch
        self.base_lr = base_lr
        self.max_iterations = max_iterations
        self.batch_size_pre_image = batch_size_per_image
        self.output_dir = output_dir
        self.score_threshold = score_threshold
        self.use_data_augmentation = use_data_augmentation
        self.annotations_json = annotations_json
        self.args = args

        self.__set_additional_params()

    def __set_additional_params(self):
        self.train_dataset_dicts = None
        self.val_dataset_dicts = None
        self.test_dataset_dicts = None
        self.dataset_dicts = None
        self.cfg = None
        self.predictor = None
        self.is_lazy = False

        self.model_name_to_path = {

            "MViTv2": {
                "mask_rcnn_mvitv2_t_3x": os.path.join(self.pretrained_models_path,
                                                      "MViTv2/configs/mask_rcnn_mvitv2_t_3x.py"),
                "cascade_mask_rcnn_mvitv2_t_3x": os.path.join(self.pretrained_models_path,
                                                              "MViTv2/configs/cascade_mask_rcnn_mvitv2_t_3x.py"),
                "cascade_mask_rcnn_mvitv2_s_3x": os.path.join(self.pretrained_models_path,
                                                              "MViTv2/configs/cascade_mask_rcnn_mvitv2_s_3x.py"),
                "cascade_mask_rcnn_mvitv2_l_in21k_lsj_50ep": os.path.join(self.pretrained_models_path,
                                                                          "MViTv2/configs/cascade_mask_rcnn_mvitv2_l_in21k_lsj_50ep.py"),
                "cascade_mask_rcnn_mvitv2_h_in21k_lsj_3x": os.path.join(self.pretrained_models_path,
                                                                        "MViTv2/configs/cascade_mask_rcnn_mvitv2_h_in21k_lsj_3x.py"),
                "cascade_mask_rcnn_mvitv2_b_in21k_3x": os.path.join(self.pretrained_models_path,
                                                                    "MViTv2/configs/cascade_mask_rcnn_mvitv2_b_in21k_3x.py"),
                "cascade_mask_rcnn_mvitv2_b_3x": os.path.join(self.pretrained_models_path,
                                                              "MViTv2/configs/cascade_mask_rcnn_mvitv2_b_3x.py"),

            },

            "ViTDet": {
                "cascade_mask_rcnn_vitdet_l_100ep": os.path.join(self.pretrained_models_path,
                                                                 "ViTDet/configs/COCO/cascade_mask_rcnn_vitdet_l_100ep.py"),
                "cascade_mask_rcnn_swin_b_in21k_50ep": os.path.join(self.pretrained_models_path,
                                                                    "ViTDet/configs/COCO/cascade_mask_rcnn_swin_b_in21k_50ep.py"),
                "mask_rcnn_vitdet_b_100ep": os.path.join(self.pretrained_models_path,
                                                         "ViTDet/configs/COCO/mask_rcnn_vitdet_b_100ep.py"),
                "cascade_mask_rcnn_swin_b_in21k_50ep2": os.path.join(self.pretrained_models_path,
                                                                     "ViTDet/configs/COCO/cascade_mask_rcnn_swin_b_in21k_50ep.py"),
                "cascade_mask_rcnn_vitdet_h_75ep": os.path.join(self.pretrained_models_path,
                                                                "ViTDet/configs/COCO/cascade_mask_rcnn_vitdet_h_75ep.py"),
                "cascade_mask_rcnn_mvitv2_b_in21k_100ep": os.path.join(self.pretrained_models_path,
                                                                       "ViTDet/configs/COCO/cascade_mask_rcnn_mvitv2_b_in21k_100ep.py")
            },

            "Transfiner": {
                "R101-FPN-DCN": f"{os.path.join(self.pretrained_models_path, 'transfiner/mask_rcnn_R_101_FPN_3x_DCN_transfiner.yaml')}",
                "R101-FPN": f"{os.path.join(self.pretrained_models_path, 'transfiner/mask_rcnn_R_101_FPN_3x_transfiner.yaml')}",
                "R50-FPN-DCN": f"{os.path.join(self.pretrained_models_path, 'transfiner/mask_rcnn_R_50_FPN_3x_DCN_transfiner.yaml')}",
                "R50-FPN": f"{os.path.join(self.pretrained_models_path, 'transfiner/mask_rcnn_R_50_FPN_3x_transfiner.yaml')}",
                "pre-trained": f"{os.path.join(self.pretrained_models_path, 'transfiner/mask_rcnn_R_101_FPN_3x_DCN_transfiner_set_16.yaml')}"
            },

            "Swin-T": {
                "V1": f"{os.path.join(self.pretrained_models_path, 'swin-transformer/swin_tiny_patch4_window7_224_22k.yaml')}",
                "T": f"{os.path.join(self.pretrained_models_path, 'swin-transformer/mask_rcnn_swint_T_FPN_3x.yaml')}",
            },

            "Swin-B": {
                "V1": f"{os.path.join(self.pretrained_models_path, 'swin-transformer/swin_base_patch4_window7_224_22k.yaml')}",
                "V2": f"{os.path.join(self.pretrained_models_path, 'swin-transformer/swinv2_base_patch4_window12_192_22k.yaml')}",
            },

            "Swin-L": {
                "V1": f"{os.path.join(self.pretrained_models_path, 'swin-transformer/swin_large_patch4_window7_224_22k.yaml')}",
                "V2": f"{os.path.join(self.pretrained_models_path, 'swin-transformer/swinv2_large_patch4_window12_192_22k.yaml')}",
            },

            "Swin-Custom": {
                "v1-t": f"{os.path.join(self.pretrained_models_path, 'swin-transformer/mask_rcnn_swint_T_FPN_3x_c.yaml')}",
                "v1-s": f"{os.path.join(self.pretrained_models_path, 'swin-transformer/mask_rcnn_swint_T_FPN_3x_c.yaml')}",
                "v1-b": f"{os.path.join(self.pretrained_models_path, 'swin-transformer/swin_base_patch4_window7_224_22k_c.yaml')}",
                "v1-l": f"{os.path.join(self.pretrained_models_path, 'swin-transformer/swin_large_patch4_window7_224_22k_c.yaml')}",
                "v1-h": f"{os.path.join(self.pretrained_models_path, 'swin-transformer/mask_rcnn_swint_T_FPN_3x_c.yaml')}",
            },

            "Mask R-CNN": {
                "R50-C4-1x": "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml",
                "R50-DC5-1x": "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x.yaml",
                "R50-FPN-1x": "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml",
                "R50-C4-3x": "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml",
                "R50-DC5-3x": "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml",
                "R50-FPN-3x": "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
                "R101-C4-3x": "COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml",
                "R101-DC5-3x": "COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml",
                "R101-FPN-3x": "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
                "X101-FPN-3x": "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml",

            },

        }

    def __set_new_hparameters(self, **kwargs):
        """
        Change some of the parameters
        Arguments:
            lr: Learning rate.
            itr: Number of iterations or epochs
            image_batch: This is also the number of training images per step.
            batch: Number of regions per image used to train RPN.
        Returns:
            model name  based on the parameters
        """
        self.__dict__.update(kwargs)
        model_name = f"model time-{time.time()}.pth"
        return model_name

    def __set_cfg(self):
        """
        Set the variables for Detectron2.
        Returns:
            Detectron2 CfgNode
        more: https://detectron2.readthedocs.io/en/latest/modules/config.html
        """

        self.__select_model()
        if not self.is_lazy:
            self.__set_advanced_cfg()

        return self.cfg

    def __set_basic_cfg(self):
        """
        Set the basic config for the traditional methods.
        """

        self.cfg.SEED = 1

        self.cfg.DATASETS.TRAIN = (self.train_set_name,)
        self.cfg.DATASETS.VAL = (self.validation_set_name,)
        self.cfg.DATASETS.TEST = (self.test_set_name,)

        self.cfg.DATALOADER.NUM_WORKERS = self.loaders_number_of_workers

        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = self.batch_size_pre_image
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.categories)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.score_threshold

        self.cfg.SOLVER.IMS_PER_BATCH = self.image_pre_batch
        self.cfg.SOLVER.BASE_LR = self.base_lr
        self.cfg.SOLVER.MAX_ITER = self.max_iterations

        if not self.use_gpu:
            self.cfg.MODEL.DEVICE = 'cpu'

        self.cfg.OUTPUT_DIR = self.output_dir
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)

    def __set_advanced_cfg(self):
        """
        Set the advanced config for the traditional methods.
        """

        self.__set_basic_cfg()

        # Backbone
        # self.cfg.MODEL.BACKBONE.FREEZE_AT = 2
        # There are 5 stages in ResNet. The first is a convolution, and the following stages are each group of residual blocks.

        # FPN
        self.cfg.MODEL.FPN.OUT_CHANNELS = 256  # power of 2

        # Types for fusing the FPN top-down and lateral features. Can be either "sum" or "avg"
        self.cfg.MODEL.FPN.FUSE_TYPE = "sum"

        # RPN
        self.cfg.MODEL.RPN.IOU_THRESHOLDS = [0.3, 0.7]
        self.cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
        self.cfg.MODEL.RPN.POSITIVE_FRACTION = 0.5
        self.cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE = "smooth_l1"
        self.cfg.MODEL.RPN.NMS_THRESH = 0.7
        self.cfg.MODEL.RPN.SMOOTH_L1_BETA = 0.0

        # ROI HEADS options
        self.cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.5]
        self.cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25
        self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.7

        # Proposal generator options
        # Current proposal generators include "RPN", "RRPN" and "PrecomputedProposals"
        # self.cfg.MODEL.PROPOSAL_GENERATOR.NAME = "RPN"

        self.cfg.SOLVER.CHECKPOINT_PERIOD = 2000
        self.cfg.SOLVER.STEPS = list(range(5000, self.max_iterations, 2000))

        # self.cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
        self.cfg.SOLVER.GAMMA = 0.97
        self.cfg.SOLVER.WARMUP_ITERS = 0

        self.cfg.INPUT.MASK_FORMAT = 'bitmask'

        if not self.use_gpu:
            self.cfg.MODEL.DEVICE = 'cpu'
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)

        # Specific test options
        # self.cfg.TEST.EXPECTED_RESULTS = []
        # self.cfg.TEST.EVAL_PERIOD = 500

        # self.cfg.freeze()

    def __set_lazy_configs(self, model_file):
        """
        Set the lazy config for the new models
        Args:
            model_file: String, path to the model file (.py)

        """
        self.args.opts = [f'dataloader.train.dataset.names={self.train_set_name}',
                          f'dataloader.test.dataset.names={self.validation_set_name}',
                          f'train.output_dir={self.output_dir}',
                          f'dataloader.train.num_workers={self.loaders_number_of_workers}',
                          f'model.proposal_generator.batch_size_per_image={self.batch_size_pre_image}',
                          f'model.roi_heads.num_classes={len(self.categories)}',
                          f'optimizer.lr={self.base_lr}',

                          f'optimizer.weight_decay={0}',
                          f'dataloader.evaluator.output_dir={self.output_dir}',
                          f'train.max_iter={self.max_iterations}',
                          f'dataloader.train.total_batch_size={self.image_pre_batch}',
                          f'model.pixel_std={[1., 1., 1.]}'
                          ]

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        LazyConfig.apply_overrides(self.cfg, self.args.opts)
        LazyConfig.save(self.cfg, "cfg.yaml")

        self.args.config_file = model_file
        self.cfg.resume = False
        self.is_lazy = True
        default_setup(self.cfg, self.args)

    def __load_dictionaries(self):
        """
        Load the data annotations
        """
        train_dataset_dicts = load_json(file_path=os.path.join(self.dictionaries_path, self.dictionaries_names[0]))
        val_dataset_dicts = load_json(file_path=os.path.join(self.dictionaries_path, self.dictionaries_names[1]))
        test_dataset_dicts = load_json(file_path=os.path.join(self.dictionaries_path, self.dictionaries_names[2]))
        dataset_dicts = load_json(file_path=os.path.join(self.dictionaries_path, self.dictionaries_names[3]))

        self.train_dataset_dicts = train_dataset_dicts
        self.val_dataset_dicts = val_dataset_dicts
        self.test_dataset_dicts = test_dataset_dicts
        self.dataset_dicts = dataset_dicts

        return train_dataset_dicts, val_dataset_dicts, dataset_dicts

    def __save_dictionaries(self):
        """
        Save the data annotations as JSON file
        """

        logging.info("Creating new data dictionaries")
        try:
            train_dataset_dicts, val_dataset_dicts, test_dataset_dicts, dataset_dicts = self.__shuffle_split_dataset()

            if not os.path.exists(self.dictionaries_path):
                os.makedirs(self.dictionaries_path)

            save_json(train_dataset_dicts, file_path=os.path.join(self.dictionaries_path, self.dictionaries_names[0]))
            save_json(val_dataset_dicts, file_path=os.path.join(self.dictionaries_path, self.dictionaries_names[1]))
            save_json(test_dataset_dicts, file_path=os.path.join(self.dictionaries_path, self.dictionaries_names[2]))
            save_json(dataset_dicts, file_path=os.path.join(self.dictionaries_path, self.dictionaries_names[3]))
            self.train_dataset_dicts = train_dataset_dicts
            self.val_dataset_dicts = val_dataset_dicts
            self.test_dataset_dicts = test_dataset_dicts
            self.dataset_dicts = dataset_dicts

        except Exception as ex:
            logging.exception(f"An exception has occurred while loading the annotation files: {ex}")
            return False

        logging.info("Data dictionaries created successfully")

        return True

    def __register_dataset(self):
        """
        Register new dataset with current detectron2 session.
        """

        for d in ["train", "val", "test"]:
            DatasetCatalog.register(f"{self.dataset_name}_" + d, lambda d=d: self.__get_data_dicts(d))
            MetadataCatalog.get(f"{self.dataset_name}_" + d).set(thing_classes=self.categories)
            MetadataCatalog.get(f"{self.dataset_name}_" + d).set(stuff_colors=[(102, 255, 102)])

        self.metadata = MetadataCatalog.get(f"{self.dataset_name}_train")
        MetadataCatalog.get(f"{self.dataset_name}_train").set(thing_classes=self.categories)

    def __find_model(self):
        """
        Select the model based on loading method.
        Returns:
            model path, and whether it's contained in the models' zoo.
        """

        if self.model_name == "Mask R-CNN":
            return self.model_name_to_path[self.model_name][self.backbone], True
        else:
            return self.model_name_to_path[self.model_name][self.backbone], False

    def __select_model(self):
        """
        Select the model based on loading method.
        """

        model_file, zoo_model = self.__find_model()

        self.cfg = get_cfg()
        self.is_lazy = False

        if zoo_model:
            self.cfg.merge_from_file(model_zoo.get_config_file(model_file))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_file)
        else:
            if self.model_name in ["ViTDet", "MViTv2"]:
                self.cfg = LazyConfig.load(model_file)
                self.__set_lazy_configs(model_file)
            else:
                self.cfg.merge_from_file(
                    model_file)

    def __load_model(self):
        """
        Load the model from the latest checkpoint
        """

        self.cfg = self.__set_cfg()
        model = build_model(self.cfg)
        DetectionCheckpointer(model).load(os.path.join(self.cfg.OUTPUT_DIR, self.model_name))
        return model

    def __lazy_test(self, cfg, model, evaluator=None):
        """
        Test the model's performance for the lazy config loader method.
        Args:
            cfg: A dictionary containing the required configs.
            model: The model to be tested
            evaluator: ...
        Returns:
            Results

        """
        if "evaluator" in cfg.dataloader:
            if evaluator is None:
                ret = inference_on_dataset(
                    model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
                )
            else:
                ret = inference_on_dataset(
                    model, instantiate(cfg.dataloader.test), evaluator
                )
            print_csv_format(ret)
            return ret

    def __lazy_train(self, args, cfg):
        """
        Args:
            cfg: an object with the following attributes:
                model: instantiate to a module
                dataloader.{train,test}: instantiate to dataloaders
                dataloader.evaluator: instantiate to evaluator for test set
                optimizer: instantaite to an optimizer
                lr_multiplier: instantiate to a fvcore scheduler
                train: other misc config defined in `configs/common/train.py`, including:
                    output_dir (str)
                    init_checkpoint (str)
                    amp.enabled (bool)
                    max_iter (int)
                    eval_period, log_period (int)
                    device (str)
                    checkpointer (dict)
                    ddp (dict)
        """
        model = instantiate(cfg.model)
        logger = logging.getLogger("detectron2")
        logger.info("Model:\n{}".format(model))
        model.to(cfg.train.device)

        cfg.optimizer.params.model = model
        optim = instantiate(cfg.optimizer)

        train_loader = instantiate(cfg.dataloader.train)

        model = create_ddp_model(model, **cfg.train.ddp)
        trainer = (AMPTrainer if cfg.train.amp.enabled else SimpleTrainer)(model, train_loader, optim)
        checkpointer = DetectionCheckpointer(
            model,
            cfg.train.output_dir,
            trainer=trainer,
        )
        trainer.register_hooks(
            [
                hooks.IterationTimer(),
                hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
                hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
                if comm.is_main_process()
                else None,
                hooks.EvalHook(cfg.train.eval_period, lambda: self.__lazy_test(cfg, model)),
                hooks.PeriodicWriter(
                    default_writers(cfg.train.output_dir, cfg.train.max_iter),
                    period=cfg.train.log_period,
                )
                if comm.is_main_process()
                else None,
            ]
        )

        checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
        if args.resume and checkpointer.has_checkpoint():
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration
            start_iter = trainer.iter + 1
        else:
            start_iter = 0
        trainer.train(start_iter, cfg.train.max_iter)

    def __shuffle_split_dataset(self, validation_split=0.1, test_split=0.1, shuffle=True, random_seed=123):
        """
        Arguments:
            validation_split: Percentage of data to use for validation.
            test_split: Percentage of data to use for testing.
            shuffle: Boolean indicating whether to shuffle the data before splitting it.
            random_seed: NumPy random seed.
        Returns:
            Dictionaries for the train, validation, and test datasets.
        """

        np.random.seed(random_seed)

        dataset_dicts = []
        for dataset in self.data_path:
            logging.info(f"Loading dataset: {dataset}")
            dataset_dicts = self.__retrieve_annotations(dataset, dataset_dicts, True)

        indices = list(np.arange(len(dataset_dicts), dtype=np.int32))

        if shuffle:
            np.random.shuffle(indices)
        dataset_dicts = np.array(dataset_dicts)[indices]

        # validation split
        training_samples_val = int(len(indices) * (1 - validation_split))
        training_samples_test = int(training_samples_val * (1 - test_split))

        val_dataset_dicts = list(dataset_dicts[training_samples_val:])
        train_dataset_dicts = list(dataset_dicts[0:training_samples_test])
        test_dataset_dicts = list(dataset_dicts[training_samples_test:training_samples_val])

        return train_dataset_dicts, val_dataset_dicts, test_dataset_dicts, dataset_dicts

    def __retrieve_annotations(self, img_dir, dataset_dicts=[], coco_format=True):
        """
        Args:
            img_dir: String, path to the directory containing the annotations.
            dataset_dicts: Dictionary, containing the dataset annotations.
        Returns:
            dataset_dicts
        """

        json_file = os.path.join(img_dir, self.annotations_json)
        if coco_format:
            # print(json_file)
            return dataset_dicts + load_json(json_file)

        filename = None
        with open(json_file) as f:
            imgs_anns = json.load(f)

        for idx, v in enumerate(imgs_anns.values()):
            try:
                record = {}

                filename = os.path.join(img_dir, v["filename"])
                height, width = cv2.imread(filename).shape[:2]

                record["file_name"] = filename
                record["image_id"] = idx
                record["height"] = height
                record["width"] = width

                annos = v["regions"]
                objs = []
                for anno in annos:
                    px = anno["x"]
                    py = anno["y"]
                    poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                    poly = [p for x in poly for p in x]

                    obj = {
                        "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "segmentation": [poly],
                        "category_id": self.categories.index(anno['object_name']),
                    }
                    objs.append(obj)
                record["annotations"] = objs
                dataset_dicts.append(record)
            except AttributeError as ae:
                logging.exception(f"An exceptions has occurred while reading the file: {filename}\n"
                                  + f"Details: {ae}")
                pass
        return dataset_dicts

    def __get_data_dicts(self, dataset="train"):
        """
        Arguments:
            dataset: Name of the dataset to take the examples from.
        Returns:
             Required dataset dictionaries
        """

        if dataset == "train":
            return self.train_dataset_dicts
        elif dataset == "val":
            return self.val_dataset_dicts
        else:
            return self.test_dataset_dicts

    def __predict(self, frame, parallel=False):
        """
        Start the inference/predictions model
        Arguments:
            frame: An array containing the image information.
            parallel: A boolean indicating whether to used multiple-GPUs or not.
        Returns:
            Detected instances.
        """

        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(self.cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(self.cfg)

        outputs = self.predictor(frame)
        instances = outputs['instances'].to("cpu")

        return instances

    def display_dataset_samples(self, dataset_dicts, number_of_examples=10):
        """
        Args:
            dataset_dicts: A dictionary containing the information needed to display the data.
            number_of_examples: Int, indicating the number of examples to display.
        """

        self.metadata = MetadataCatalog.get(self.train_set_name)

        for d in random.sample(dataset_dicts, number_of_examples):
            img = cv2.imread(d["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata=self.metadata, scale=0.5)
            out = visualizer.draw_dataset_dict(d)
            cv2.imshow("Segmentation Example", out.get_image()[:, :, ::-1])
            cv2.waitKey()

    def display_examples(self, dataset="test", number_of_examples=10):
        """
        Display few examples for testing purposes.
        Arguments:
            dataset: Name of the dataset to take the examples from.
            number_of_examples: Int, indicating the number of examples to display.
        """

        dataset_dicts = self.__get_data_dicts(dataset)
        for d in random.sample(dataset_dicts, number_of_examples):
            im = cv2.imread(d["file_name"])
            outputs = self.__predict(
                im)
            v = Visualizer(im[:, :, ::-1],
                           metadata=self.metadata,
                           scale=1.,
                           )
            out = v.draw_instance_predictions(outputs)
            cv2.imshow("Segmentation Example", out.get_image()[:, :, ::-1])
            cv2.waitKey()

    def train(self):
        """
        Method to load and retrain the pre-trained model
        """

        if check_files(self.dictionaries_names,
                       self.dictionaries_path) and self.load_pre_existing_dicts:
            self.__load_dictionaries()
        else:
            self.__save_dictionaries()

        logging.info(f"Number of examples in the training set: {len(self.train_dataset_dicts)}")
        logging.info(f"Number of examples in the validation set: {len(self.val_dataset_dicts)}")
        logging.info(f"Number of examples in the test set: {len(self.test_dataset_dicts)}")

        try:
            self.__register_dataset()
        except Exception as ex:
            logging.exception(f"An exception has occurred while processing the dataset json dictionaries: {ex}")
        finally:
            self.cfg = self.__set_cfg()

            if not self.is_lazy:
                if self.use_data_augmentation:
                    self.trainer = CustomAugTrainer(self.cfg)
                    self.trainer.register_hooks(
                        [hooks.EvalHook(0, lambda: self.trainer.test_with_TTA(self.cfg, self.trainer.model))]
                    )
                else:
                    self.trainer = DefaultTrainer(self.cfg)

                # DefaultTrainer.test(self.cfg, self.trainer.model, evaluators=COCOEvaluator(self.test_set_name, ("bbox", "segm"), True, output_dir=self.cfg.OUTPUT_DIR,
                #                       use_fast_impl=False))

                val_loss = ValidationLoss(self.cfg)
                self.trainer.register_hooks([val_loss])
                self.trainer._hooks = self.trainer._hooks[:-2] + self.trainer._hooks[-2:][
                                                                 ::-1]  # swap the order of PeriodicWriter and ValidationLoss
                self.trainer.resume_or_load(resume=True)

                return self.trainer.train()
            else:
                self.__lazy_train(self.args, self.cfg)

    def k_folds(self, cv=5, split=None, shuffle=True, random_seed=123):
        """
        Only cv or split should be set, the other one will be calculated automatically.
        In order to get Leave-One-Out LOOCV is when cv = N (number of examples) so there is no need
        to shuffle the data.
        Arguments:
            cv: Number of k-folds to be used.
            split: Percentage of data to use for testing.
            shuffle: Boolean indicating whether to shuffle the data or not.
            random_seed: NumPy random seed.
        Returns:
            Generators for the test_dataset_dicts, train_dataset_dicts the number of yielded generators
            is equal to the number of cv.
        """

        np.random.seed(random_seed)
        dataset_dicts = self.__retrieve_annotations()
        if split and cv is None:
            logging.exception("Neither cv nor split ration are defined")
            raise Exception("Neither cv nor split ration are defined")
        if split and cv is not None:
            logging.exception("Both cv and split are defined")
            raise Exception("Both cv and split are defined")

        if cv is None:
            cv = round(1. / split)
            split = 1. / cv
        elif split is None and cv is not None:
            split = 1. / cv

        for i in range(cv):

            indices = list(np.arange(len(dataset_dicts), dtype=np.int32))
            if shuffle:
                np.random.shuffle(indices)

            dataset_dicts = np.array(dataset_dicts)[indices]

            test_samples = int(len(indices) * split)
            test_dataset_dicts = list(dataset_dicts[i * test_samples:test_samples + (i * test_samples)])
            train_dataset_dicts = list(dataset_dicts[list(
                set(range(len(indices))) -
                set(range(i * test_samples, test_samples + (i * test_samples)))
            )])
            yield test_dataset_dicts, train_dataset_dicts

    def val(self):
        """
        Validate the dataset using the conventional method
        Returns:
        """

        model = CustomTrainer.build_model(self.cfg)
        DetectionCheckpointer(model, save_dir=self.cfg.OUTPUT_DIR).resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=True
        )
        res = CustomTrainer.test(self.cfg, model)
        if self.cfg.TEST.AUG.ENABLED:
            res.update(CustomTrainer.test_with_TTA(self.cfg, model))
        if comm.is_main_process():
            verify_results(self.cfg, res)
        return res

    def validate(self):
        """
        Evaluate the model on the test dataset
        Returns Validation results
        """
        if self.is_lazy:
            model = instantiate(self.cfg.model)
            model.to(self.cfg.train.device)
            model = create_ddp_model(model)
            DetectionCheckpointer(model).load(self.cfg.train.init_checkpoint)
            evaluator = COCOEvaluator(self.test_set_name, ("bbox", "segm"), True, output_dir=self.output_dir,
                                      use_fast_impl=False)
            validation_results = self.__lazy_test(self.cfg, model, None)

            print(validation_results)
        else:
            evaluator = COCOEvaluator(self.test_set_name, ("bbox", "segm"), True, output_dir=self.output_dir,
                                      use_fast_impl=False)
            val_loader = build_detection_test_loader(self.cfg, self.test_set_name)
            validation_results = inference_on_dataset(self.trainer.model, val_loader, evaluator)

        return validation_results
