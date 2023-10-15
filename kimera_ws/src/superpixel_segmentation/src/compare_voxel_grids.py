from voxelgrid import VoxelGrid
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from collections import defaultdict
import sys
from typing import Dict, List, Tuple

# TODO CENTRAILSE CLASS COLOR MAPPING DEFINITION TO ONE PYTHON FILE
#CLASS_NAMES = ['BG', 'bed', 'books', 'ceiling', 'chair', 'floor',
#                'furniture', 'objects', 'picture', 'sofa', 'table',
#                'tv', 'wall', 'window']
#
#CLASS_COLORS = [(0, 0, 0), (119, 119, 119), (244, 243, 131),
#                (137, 28, 157), (150, 255, 255), (54, 114, 113),
#                (0, 0, 176), (255, 69, 0), (87, 112, 255), (0, 163, 33),
#                (255, 150, 255), (255, 180, 10), (101, 70, 86),
#                (38, 230, 0)]


CLASS_NAMES = [
    'wall',
    'floor',
    'ceiling',
    'chair',
    'table',
    'window',
    'curtain',
    'picture',
    'bed',
    'sofa',
    #'pillow',
    #'monitor',
    #'sink',
    #'trashcan',
    #'toilet',
    #'refrigerator',
    #'oven',
    #'bathtub',
    #'cabinet',
    #'object'
]

CLASS_COLORS = [
    (119,119,119),
    (244,243,131),
    (255,190,190),
    (54,114,113),
    (255,150,255),
    (0,163,33),
    (150,255,0),
    (255,180,10),
    (150,255,255),
    (0,0,176),
    #(24,209,255),
    #(152,163,55),
    #(70,72,115),
    #(87,64,34),
    #(193,195,234),
    #(192,79,212),
    #(70,72,115),
    #(52,57,131),
    #(137,28,157), 
    #(255,69,0)
]



CLASS_TO_COL: Dict[str, Tuple[int,int,int]] = dict(zip(CLASS_NAMES, CLASS_COLORS))
COL_TO_CLASS: Dict[Tuple[int,int,int], str] = dict(zip(CLASS_COLORS, CLASS_NAMES))
COL_TO_IND: Dict[Tuple[int,int,int], int] = dict(zip(CLASS_COLORS, list(range(len(CLASS_COLORS)))))


def create_name(pth:str):
    return "/".join(pth.split("/")[-2:])

def compare(pths: List[str], confusion_mtx=False):
    assert len(pths) >= 3
    left_pth = pths[1]

    print("Evaluating paths:")
    for p in pths[2:]:
        print(create_name(p))

    print()
    print("<", create_name(left_pth))
    print()
    
    left = VoxelGrid(left_pth)

    for right_pth in pths[2:]:
        print(">", create_name(right_pth))
        right = VoxelGrid(right_pth)
        evaluator = Evaluator(left, right, "restrict")
        jac = evaluator.jaccard()
        sor = evaluator.sorensen_dice()
        acc = evaluator.color_accuracy()
        print(f"Jaccard\t\t\tSorensen\t\tColor-acc")
        print(f"{jac}\t{sor}\t{acc}")
        print()

        if confusion_mtx:
            cnf = evaluator.confusion_mat()
            disp = ConfusionMatrixDisplay((cnf / cnf.sum() * 100).round(1), display_labels=CLASS_NAMES)
            disp.plot()
            plt.xticks(rotation=90)
            name = right_pth.split("/")[-1]
            plt.savefig(f"../../../../notes_and_logs/cnf_mtxs/{name}.png", dpi=200)


class Evaluator:
    def __init__(self, left: VoxelGrid, right: VoxelGrid, cat_mode: str = "all"):
        self.left = left
        self.right = right
        self.cat_mode = cat_mode

    def jaccard(self) -> float:
        return self.color_intersection() / self.union()

    def sorensen_dice(self) -> float:
        return 2 * (self.color_intersection()) / (self.cardinallity(self.left) + self.cardinallity(self.right))

    def color_accuracy(self) -> float:
        return self.color_intersection() / self.voxel_intersection()

    def union(self) -> int:
        c = 0
        left_keys = set(self.left.voxels.keys())
        right_keys = set(self.right.voxels.keys())
        if self.cat_mode == "all":
            c = len(left_keys | right_keys)
        elif self.cat_mode == "restrict":
            for vox in left_keys:
                if self.left.get_color_voxel(vox) in COL_TO_CLASS:
                    c += 1
            for vox in right_keys - left_keys:
                if self.right.get_color_voxel(vox) in COL_TO_CLASS:
                    c += 1
        else:
            for vox in left_keys:
                if COL_TO_CLASS.get(self.left.get_color_voxel(vox), "Not") == self.cat_mode:
                    c += 1
            for vox in right_keys - left_keys:
                if COL_TO_CLASS.get(self.left.get_color_voxel(vox), "Not") == self.cat_mode:
                    c += 1
        return c

    def cardinallity(self, vg: VoxelGrid) -> int:
        c = 0
        if self.cat_mode == "all":
            c = len(vg.voxels.keys())
        elif self.cat_mode == "restrict":
            for vox in vg.voxels.keys():
                if vg.get_color_voxel(vox) in COL_TO_CLASS:
                    c += 1
        else:
            for vox in vg.voxels.keys():
                if COL_TO_CLASS.get(self.left.get_color_voxel(vox), "Not") == self.cat_mode:
                    c += 1
        return c

    def voxel_intersection(self) -> int:
        c = 0
        if self.cat_mode == "all":
            for vox in self.left.voxels:
                if vox in self.right.voxels:
                    c += 1
        elif self.cat_mode == "restrict":
            for vox in self.left.voxels:
                if self.left.get_color_voxel(vox) in COL_TO_CLASS:
                    if vox in self.right.voxels and self.right.get_color_voxel(vox) in COL_TO_CLASS:
                        c += 1
        else:
            print("Voxel intersection is not defined for a single category")
            c = 1
        return c

    def color_intersection(self) -> int:
        c = 0
        if self.cat_mode == "all":
            for voxel in self.left.voxels:
                if voxel in self.right.voxels:
                    if self.left.get_color_voxel(voxel) == self.right.get_color_voxel(voxel):
                        c += 1
        elif self.cat_mode == "restrict":
            for voxel in self.left.voxels:
                if voxel in self.right.voxels:
                    if (self.left.get_color_voxel(voxel) in COL_TO_CLASS and
                        self.left.get_color_voxel(voxel) == self.right.get_color_voxel(voxel)):
                        c += 1
        else:
            print("Color intersection is not defined for a single category")
            c = 1
        return c

    def confusion_mat(self) -> np.ndarray:
        result = np.zeros((len(CLASS_COLORS), len(CLASS_COLORS)))
        for voxel in self.left.voxels:
            if voxel in self.right.voxels:
                gt_ind = COL_TO_IND.get(self.left.get_color_voxel(voxel), -1)
                pr_ind = COL_TO_IND.get(self.right.get_color_voxel(voxel), -1)
                if gt_ind != -1 and pr_ind != -1:
                    result[gt_ind, pr_ind] += 1

        return result

if __name__=='__main__':
    compare(sys.argv)