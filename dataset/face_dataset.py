import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import os
import cv2
from hp_visualize import resize

class AFLW2000Dataset(Dataset):
    def __init__(self,data_path):
        super(AFLW2000Dataset, self).__init__()


    def __len__(self):
        pass

    def __getitem__(self,idx):
        pass

class WLPDataset(Dataset):
    def __init__(self,annotate_path, input_shape):
        super(WLPDataset, self).__init__()
        with open(annotate_path, 'r') as f:
            lines = f.readlines()
        self.lines = lines
        self.labels = np.ones((len(self.lines), 1))
        self.input_shape = input_shape

    def __len__(self):
        return len(self.lines)


    def __getitem__(self,idx):
        line = self.lines[idx]
        line = line.strip().split(' ', 1)
        img_path = line[0].replace("/content/data/data/", "data/300W_LP_AFLW2000/")
        bbox_attr = line[1].split(' ')
        bbox_attr = [float(x) for x in bbox_attr]
        xmin, ymin, xmax, ymax = bbox_attr[:4]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image[int(ymin):int(ymax), int(xmin):int(xmax)]
        image = cv2.resize(image, tuple(self.input_shape[:2]))/255.0
        image = torch.from_numpy(image)

        label = self.labels[idx]

        return image, label


