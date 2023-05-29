import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from Inference.utils.predictor import VisualizationDemo
from PIL import Image

WINDOW_NAME = "instance segmentation"

categories = ['wall', 'floor', 'ceiling', 'chair', 'table', 'window', 'curtain',
              'picture', 'bed', 'sofa', 'pillow', 'monitor',
              'sink', 'trashcan', 'toilet', 'refrigerator', 'oven', 'bathtub', 'cabinet', 'object']


def setup_cfg(args):
    """
    Set up the Detectron2 configuration from command-line arguments.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        cfg (CfgNode): The Detectron2 configuration.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(categories)
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    """
    Create an argument parser for command-line arguments.

    Returns:
        parser (argparse.ArgumentParser): The argument parser.
    """
    parser = argparse.ArgumentParser(description="Detectron2 inference")
    parser.add_argument(
        "--config-file",
        default="configs/mask_rcnn_R_101_FPN_3x_transfiner_deform.yaml",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
    )

    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
             "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.8,
        help="Minimum score for instance predictions to be shown",
    )

    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )

    return parser


def test_opencv_video_format(codec, file_ext):
    """
    Test if the specified video codec and file extension are supported by OpenCV.

    Args:
        codec (str): The video codec.
        file_ext (str): The file extension.

    Returns:
        supported (bool): True if the video format is supported, False otherwise.
    """
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg, categories)

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"

        for path in tqdm.tqdm(args.input, disable=not args.output):
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)

            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))

                else:
                    os.makedirs(args.output)
                    out_filename = os.path.join(args.output, os.path.basename(path))

                im = Image.fromarray(visualized_output)
                im.save(out_filename)

            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])

                if cv2.waitKey(0) == 27:
                    break
