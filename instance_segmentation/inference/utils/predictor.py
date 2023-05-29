import atexit
import bisect
import multiprocessing as mp
import numpy as np
import torch

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
import matplotlib.pyplot as plt

from file_writer import save_json


class VisualizationDemo(object):
    def __init__(self, cfg, categories, instance_mode=ColorMode.IMAGE, parallel=False):
        """Visualization demo for object detection.

        Args:
            cfg (CfgNode): Configuration node for the model.
            categories (list): List of category names.
            instance_mode (ColorMode): Instance visualization mode.
            parallel (bool): Whether to run the model in parallel processes.
        """
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused")
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode
        self.categories = categories
        self.parallel = parallel

        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)

        cmap = plt.get_cmap('viridis')
        self.colors = (np.delete(cmap(np.random.uniform(0, 1, len(self.categories))), -1, 1) * 255).astype(np.uint8)
        save_json(self.colors, '../colors.json')

    def run_on_image(self, image):
        """Run object detection on the given image and visualize the results.

        Args:
            image (np.ndarray): Image array of shape (H, W, C) in BGR order.

        Returns:
            predictions (dict): Model predictions.
            vis_output (VisImage): Visualized image output.
        """
        predictions = self.predictor(image)
        image_width, image_height, image_channels = image.shape

        if "instances" in predictions:
            instances = predictions["instances"].to(self.cpu_device)

            pred_masks = instances.pred_masks.numpy()
            pred_classes = instances.pred_classes.numpy()

            sorted_indices = np.argsort(pred_classes)
            pred_masks = pred_masks[sorted_indices]
            pred_classes = pred_classes[sorted_indices]
            vis_output = np.zeros(shape=(image_width, image_height, image_channels), dtype=np.uint8)

            for pred_class, pred_mask in zip(pred_classes, pred_masks):
                vis_output[pred_mask] = self.colors[pred_class]

        return predictions, vis_output


class AsyncPredictor:
    """Asynchronous predictor for running the model in parallel processes."""

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
        """Initialize the AsyncPredictor.

        Args:
            cfg (CfgNode): Configuration node for the model.
            num_gpus (int): Number of GPUs to use.
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []

        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue))

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()

        atexit.register(self.shutdown)

    def put(self, image):
        """Put an image into the task queue for processing.

        Args:
            image: Image to process.
        """
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        """Get the processed result from the result queue.

        Returns:
            The processed result.
        """
        self.get_idx += 1

        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        """Get the length of the predictor.

        Returns:
            The length of the predictor.
        """
        return self.put_idx - self.get_idx

    def __call__(self, image):
        """Call the predictor on the given image.

        Args:
            image: Image to process.

        Returns:
            The processed result.
        """
        self.put(image)
        return self.get()

    def shutdown(self):
        """Shutdown the predictor and stop the worker processes."""
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        """Get the default buffer size.

        Returns:
            The default buffer size.
        """
        return len(self.procs) * 5
