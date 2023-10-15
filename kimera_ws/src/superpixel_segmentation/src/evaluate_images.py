import numpy as np
import sys
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from numba import guvectorize, uint16
from tqdm import tqdm

import cv2
from sklearn import metrics

from matplotlib import pyplot as plt


@guvectorize([(uint16[:, :, :], uint16[:, :])], "(h,w,c)->(h,w)", nopython=True)
def transform_class_colors_to_numbering(img, result):
    # compile time needed, may seek a better way by passing some parameter
    col_to_ind = {
        (0, 0, 0): 0,
        (119, 119, 119): 1,
        (244, 243, 131): 2,
        (137, 28, 157): 3,
        (150, 255, 255): 4,
        (54, 114, 113): 5,
        (0, 0, 176): 6,
        (255, 69, 0): 7,
        (87, 112, 255): 8,
        (0, 163, 33): 9,
        (255, 150, 255): 10,
        (255, 180, 10): 11,
        (101, 70, 86): 12,
        (38, 230, 0): 13,
    }
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            b, g, r = img[i, j, :]
            result[i, j] = col_to_ind[(r, g, b)]


class ImageDataset(Dataset):
    def __init__(self, data_root: Path, sample_size: int = -1):
        seman_folder = data_root / "seman"
        mrcnn_folder = data_root / "mrcnn"
        supix_folder = data_root / "supix"

        self.gt_list = np.array(list(seman_folder.glob("*.png")))
        self.mrcnn_list = np.array(list(mrcnn_folder.glob("*.png")))
        self.supix_list = np.array(list(supix_folder.glob("*.png")))

        if sample_size != -1:
            mask = np.random.choice(len(self.gt_list), sample_size)
            self.gt_list = self.gt_list[mask]
            self.mrcnn_list = self.mrcnn_list[mask]
            self.supix_list = self.supix_list[mask]

    def __len__(self):
        return len(list(self.gt_list))

    def __getitem__(self, idx):
        gt_image = cv2.imread(str(self.gt_list[idx]))
        mrcnn_image = cv2.imread(str(self.mrcnn_list[idx]))
        supix_image = cv2.imread(str(self.supix_list[idx]))

        gt = np.zeros((gt_image.shape[0], gt_image.shape[1]))
        mrcnn = np.zeros((mrcnn_image.shape[0], mrcnn_image.shape[1]))
        supix = np.zeros((supix_image.shape[0], supix_image.shape[1]))

        transform_class_colors_to_numbering(gt_image, gt)
        transform_class_colors_to_numbering(mrcnn_image, mrcnn)
        transform_class_colors_to_numbering(supix_image, supix)

        return calculate_metrics(gt.ravel(), mrcnn.ravel(), supix.ravel())


def calculate_metrics(gt: np.ndarray, mrcnn: np.ndarray, supix: np.ndarray):
    return calc_met(gt, mrcnn), calc_met(gt, supix)


def calc_met(gt: np.ndarray, pred: np.ndarray):
    jaccard_avg = metrics.jaccard_score(gt, pred, average="micro")
    f1 = metrics.f1_score(gt, pred, average="micro")
    #accuracy = metrics.accuracy_score(gt, pred)
    return jaccard_avg, f1#, accuracy


def main(data_path: str, sample_size: int = 20):
    data_root = Path(data_path)
    assert data_root.is_dir(), f"Provided directory does no exist: {str(data_root)}"

    dataset = ImageDataset(data_root, sample_size)
    batch_size = 16
    loader = DataLoader(dataset, num_workers=48, batch_size=batch_size)

    mrcnn_jaccard_avg = []
    mrcnn_f1 = []
    mrcnn_accuracy = []

    supix_jaccard_avg = []
    supix_f1 = []
    supix_accuracy = []

    for scores in tqdm(loader):
        mrcnn_jaccard_avg += list(scores[0][0].numpy())
        mrcnn_f1 += list(scores[0][1].numpy())
        #mrcnn_accuracy += list(scores[0][2].numpy())

        supix_jaccard_avg += list(scores[1][0].numpy())
        supix_f1 += list(scores[1][1].numpy())
        #supix_accuracy += list(scores[1][2].numpy())

    mrcnn_jaccard_avg = np.array(mrcnn_jaccard_avg)
    mrcnn_f1 = np.array(mrcnn_f1)
    mrcnn_accuracy = np.array(mrcnn_accuracy)

    supix_jaccard_avg = np.array(supix_jaccard_avg)
    supix_f1 = np.array(supix_f1)
    supix_accuracy = np.array(supix_accuracy)

    print(f"{mrcnn_jaccard_avg.mean()}\t{mrcnn_jaccard_avg.std()}")
    print(f"{mrcnn_f1.mean()}\t{mrcnn_f1.var()}")
    print()
    print(f"{supix_jaccard_avg.mean()}\t{supix_jaccard_avg.std()}")
    print(f"{supix_f1.mean()}\t{supix_f1.std()}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
