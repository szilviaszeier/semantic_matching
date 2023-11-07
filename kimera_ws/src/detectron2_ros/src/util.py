
from detectron2.config import get_cfg


categories = ['wall', 'floor', 'ceiling', 'chair', 'table', 
                'window', 'curtain', 'picture', 'bed', 'sofa',
                'pillow', 'monitor', 'sink', 'trashcan', 'toilet', 
                'refrigerator', 'oven', 'bathtub', 'cabinet',
                'object']

category_colors = [[119,119,119], [131,243,244], [190,190,255], [113,114,54], [255,150,255],
                    [33,163,0], [0,255,150], [10,180,255], [255,255,150], [176,0,0],
                    [255,209,24], [55,163,152], [115,72,70], [34,64,87], [234,195,193],
                    [212,79,192], [115,72,70], [131,57,52], [157,28,137],
                    [0,69,255]]

def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(args["config_file"])
    cfg.merge_from_list(["MODEL.WEIGHTS",args["model_path"]])
    #cfg.merge_from_list(args["opts"])
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(categories)
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args["confidence_threshold"]
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args["confidence_threshold"]
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args["confidence_threshold"]
    cfg.freeze()
    return cfg
