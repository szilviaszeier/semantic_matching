import atexit
import bisect
import copy
import json
import logging
import multiprocessing as mp
import os
from collections import OrderedDict

import detectron2.utils.comm as comm
import numpy as np
import torch
from detectron2.data import build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.engine import HookBase
from detectron2.evaluation import COCOEvaluator
from detectron2.modeling import GeneralizedRCNNWithTTA

from utils.logging_settings import setup_loging_config

setup_loging_config()


def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """

    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator = COCOEvaluator(dataset_name, ("bbox", "segm"), output_dir=output_folder,
                              use_fast_impl=False)

    return evaluator


def grid_search(**kwargs):
    """ Search a giving parameters space by looping over all the possible combinations
    Args:
        **kwargs: keyword arguments, representing the possible parameters to be fine-tuned,
            the names of these parameters should correspond to existing variables in the Trainer class
            otherwise they will go unused.

    Yields:
        A dictionary containing the new keyword arguments to be used in the training process.
    """

    keys, values = list(kwargs.keys()), list(kwargs.values())
    indices = {key: len(value) for key, value in zip(keys, values)}

    while True:
        i = 0
        nkwargs = {}

        for key in keys:
            nkwargs[key] = kwargs[key][indices[key] - 1]

            if keys[i] == key:
                indices[keys[i]] -= 1

            if indices[key] == 0:
                indices[key] = len(kwargs[key])
                i += 1

        yield nkwargs

        if i >= len(keys):
            break


def save_json(data, file_path):
    """ Save data dictionaries
    Args:
        data: A dictionary containing the dataset annotations.
        file_path: A string indicating the location and the name of json file to be saved.
    """
    with open(file_path, 'w') as outfile:
        json.dump(data, outfile, indent=4, cls=NpEncoder)


def load_json(file_path):
    """ Save data dictionaries
    Args:
        file_path: A string indicating the location and the name of json file to be loaded.
    """
    with open(file_path) as json_file:
        data = json.load(json_file)
    return data


def check_files(files, dictionaries_path):
    """ Check if the data dictionaries are saved.
    Args:
        files: The names of the data dictionary json files.
        dictionaries_path: The location of the data dictionaries.

    Returns:
        A boolean indicating whether a files exist.
    """
    for file in files:
        if not os.path.exists(os.path.join(dictionaries_path, file)):
            return False
    return True


def custom_mapper(dataset_dict):
    """ A custom data augmentation file based on https://detectron2.readthedocs.io/en/latest/tutorials/augmentation.html
    Args:
        dataset_dict: A dictionary containing the dataset annotations.

    Returns:

    """
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict["file_name"])
    width, height, _ = image.shape
    # TransformList Maintain a list of transform operations which will be applied in sequence. .. attribute:: transforms

    t_list = [
        T.RandomApply(T.RandomBrightness(intensity_min=0.8, intensity_max=1.2),
                      prob=0.1),

        T.RandomApply(T.RandomContrast(intensity_min=0.8, intensity_max=1.2),
                      prob=0.1),

        T.RandomApply(T.RandomCrop(crop_type='relative_range', crop_size=(0.7, 0.7)),
                      prob=0.1),

        T.RandomApply(T.RandomExtent(scale_range=(0.7, 0.7), shift_range=(0.7, 0.7)),
                      prob=0.1),

        T.RandomApply(T.RandomFlip(prob=1.0, horizontal=True, vertical=False),
                      prob=0.1),

        T.RandomApply(T.RandomSaturation(intensity_min=0.8, intensity_max=1.2),
                      prob=0.1),

        T.RandomApply(T.RandomLighting(scale=0.2),
                      prob=0.1),

        T.RandomApply(T.RandomRotation(angle=[-30, 30], expand=True, center=None, sample_style="range", interp=None),
                      prob=0.1),

        T.RandomApply(T.RandomRotation(angle=[-30, 30], expand=False),
                      prob=0.1),
    ]

    image, transforms = T.apply_transform_gens(t_list, image)
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
    T.Augmentation()
    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict


class CustomAugTrainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


class CustomTrainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        # return super.__base__()
        return build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


class ValidationLoss(HookBase):
    """
    Custom class to handel the validation set loss
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()
        self.cfg.DATASETS.TRAIN = cfg.DATASETS.VAL
        self._loader = iter(build_detection_train_loader(self.cfg))

    def after_step(self):
        data = next(self._loader)
        with torch.no_grad():
            loss_dict = self.trainer.model(data)

            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {"val_" + k: v.item() for k, v in
                                 comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                self.trainer.storage.put_scalars(total_val_loss=losses_reduced,
                                                 **loss_dict_reduced)


class NpEncoder(json.JSONEncoder):
    """
    Helper class to handel json data.
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput a little bit when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5
