import torch
from torch.utils.data.dataset import Dataset
import albumentations as albu
import numpy as np
import os
import cv2
from hp_visualize import convertAngleToVector, resize


class BIWIDataset(Dataset):
    def __init__(self,annotate_path, input_shape):
        super(BIWIDataset, self).__init__()
        with open(annotate_path, 'r') as f:
            lines = f.readlines()
        self.lines = lines
        self.input_shape = input_shape

    def __len__(self):
        return len(self.lines)

    def __getitem__(self,idx):
        line = self.lines[idx]
        line = line.strip().split(' ', 1)
        image_path = line[0]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bbox_attr = line[1].split(' ')
        bbox_attr = np.array([float(x) for x in bbox_attr], dtype=np.float32)
        xmin, ymin, xmax, ymax = bbox_attr[:4]
        face_img = image[ymin:ymax, xmin:xmax]
        face_img = cv2.resize(face_img, self.input_shape[:2])/255.0
        face_img = torch.from_numpy(face_img)

        yaw, pitch, roll = bbox_attr[4:]
        rotateMatrix = convertAngleToVector(yaw, pitch, roll)
        rotateMatrix = torch.from_numpy(rotateMatrix)
        return face_img, rotateMatrix

