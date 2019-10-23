"""
This is code for evaluating Segmentation.
You must specify the prediction and ground-truth segmentation map folder properly.
The final
"""

import os
import sys
import cv2
import numpy as np
from multiprocessing import Pool
np.set_printoptions(threshold=sys.maxsize)


class ConfusionMatrix(object):
    def __init__(self, nclass, classes=None):
        self.nclass = nclass
        self.classes = classes
        self.M = np.zeros((nclass, nclass))

    def add(self, gt, pred):
        assert (np.max(pred) <= self.nclass)
        assert (len(gt) == len(pred))
        for i in range(len(gt)):
            if not gt[i] == 255:
                self.M[gt[i], pred[i]] += 1.0

    def update_matrix(self, matrix):
        assert (matrix.shape == self.M.shape)
        self.M += matrix

    def __str__(self):
        pass

    def recall(self):
        recall = 0.0
        for i in range(self.nclass):
            recall += self.M[i, i] / np.sum(self.M[:, i])
        return recall / self.nclass

    def accuracy(self):
        accuracy = 0.0
        for i in range(self.nclass):
            accuracy += self.M[i, i] / np.sum(self.M[i, :])
        return accuracy / self.nclass

    def jaccard(self):
        seg_acc_per_class = []
        for i in range(self.nclass):
            if not self.M[i, i] == 0:
                seg_acc_per_class.append(self.M[i, i] / (np.sum(self.M[i, :]) + np.sum(self.M[:, i]) - self.M[i, i]))
        return np.sum(seg_acc_per_class) / len(seg_acc_per_class), seg_acc_per_class, self.M

    def calculate_item(self, item):
        gt, pred = item
        m = np.zeros((self.nclass, self.nclass))
        assert (len(gt) == len(pred))
        for i in range(len(gt)):
            if gt[i] < self.nclass:
                m[gt[i], pred[i]] += 1.0
        return m


def main(img_list_path, gt_root, pred_root):
    img_list = list()

    # Load Image List
    with open(img_list_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            info = line.strip().split()
            img_list.append(info[0].strip().split('/')[-1][:-4])

    data_list = list()
    for i, name in enumerate(img_list):
        gt_img_path = os.path.join(gt_root, name+'.png')
        pred_img_path = os.path.join(pred_root, name+'.png')

        gt_image = cv2.imread(gt_img_path, cv2.IMREAD_GRAYSCALE)
        pred_image = cv2.imread(pred_img_path, cv2.IMREAD_GRAYSCALE)

        data_list.append([gt_image.flatten(), pred_image.flatten()])

    conf_mat = ConfusionMatrix(21)
    f = conf_mat.calculate_item
    pool = Pool(16)
    m_list = pool.map(f, data_list)
    pool.close()
    pool.join()

    for m in m_list:
        conf_mat.update_matrix(m)

    mean_seg_acc, class_seg_acc, seg_matrix = conf_mat.jaccard()
    print(mean_seg_acc)
    print(class_seg_acc)
    print(seg_matrix)

if __name__ == '__main__':

    img_list = 'datalist/PascalVOC/val.txt'
    gt_path = '../../PascalVOC/VOCdevkit/VOC2012/SegmentationClassAug'
    pred_path = 'train_log/test1/final_map'

    main(img_list, gt_path, pred_path)