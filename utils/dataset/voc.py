import numpy as np
import torch
from torch.utils.data import Dataset
import PIL.Image
import os.path

IMG_FOLDER_NAME = "JPEGImages"

CAT_LIST = ['aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train',
            'tvmonitor']

MODE = ['train', 'val', 'test']

CAT_NAME_TO_NUM = dict(zip(CAT_LIST, range(len(CAT_LIST))))


class VOCDataset(Dataset):

    def __init__(self, root=None, datalist=None, transform=None, mode='train'):
        if mode not in MODE:
            raise Exception("Mode has to be one of ", MODE)
        self.root = root
        datalist = open(datalist).read().splitlines()
        self.image_names = [img_gt_name.split(' ')[0][-15:-4] for img_gt_name in datalist]
        self.transform = transform
        if mode is not 'test':
            self.label_list = load_image_label_list_from_npy(self.image_names)
        self.mode = mode
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]

        image_path = os.path.join(self.root, IMG_FOLDER_NAME, name + '.jpg')
        img = PIL.Image.open(image_path).convert("RGB")
        if self.mode is not 'test':
            label = torch.from_numpy(self.label_list[idx])
        else:
            label = None
        if self.transform:
            img = self.transform(img)

        if self.mode == 'train':
            return img, label
        else:
            return img, label, name


def load_image_label_list_from_npy(img_name_list):
    cls_labels_dict = np.load('datalist/PascalVOC/cls_labels.npy', allow_pickle=True).item()
    return [cls_labels_dict[img_name] for img_name in img_name_list]


def load_img_cue_name_list(dataset_path):
    img_cue_name_list = open(dataset_path).read().splitlines()
    img_name_list = [img_cue_name.split(' ')[0][-15:-4] for img_cue_name in img_cue_name_list]
    cue_name_list = [img_cue_name.split(' ')[1].strip() for img_cue_name in img_cue_name_list]

    return img_name_list, cue_name_list