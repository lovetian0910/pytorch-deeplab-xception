from os.path import split
from torch.utils.data import Dataset
import torch
import numpy as np
from PIL import Image
import cv2

root_path = "QGameData/"
file_path = "live_more_"
nClasses = 5
class QGameSegmentation(Dataset):
    def __init__(self, width, height, split):
        self.width = width
        self.height = height
        f = open(root_path + file_path + split + ".txt", 'r')
        self.items = f.readlines()

    def __getitem__(self, idx):
        item = self.items[idx]
        image_path = root_path + item.split(' ')[0]
        label_path = root_path + item.split(' ')[-1].strip()
        img = cv2.imread(image_path, 1)
        label_img = cv2.imread(label_path, 1)
        img = cv2.resize(img, (self.width, self.height))
        label_img = cv2.resize(label_img, (self.width, self.height))
        im = img
        lim = label_img
        lim = lim[:, :, 0]
        im = np.float32(im) / 127.5 - 1
        im = np.transpose(im, [2, 0, 1])
        torch_img = torch.from_numpy(im).float()
        mask = torch.from_numpy(lim).float()
        return {'image': torch_img, 'label': mask}

    def __len__(self):
        # return len(self.items)
        return 32