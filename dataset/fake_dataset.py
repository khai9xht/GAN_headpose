import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from hp_visualize import convertAngleToVector

class FakeDataset(Dataset):
    def __init__(self, len_dataset, vector_dim):
        super(FakeDataset, self).__init__()
        self.len_dataset = len_dataset
        self.random_pose = np.random.rand(len_dataset, vector_dim)*np.pi - np.pi/2
        self.random_pose[:, 0]  = self.random_pose[:, 0] * 2
        self.fake_labels = np.zeros((len_dataset, 1))
    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        yaw, pitch, roll = self.random_pose[idx]
        random_vector = convertAngleToVector(yaw, pitch, roll)
        X = torch.from_numpy(random_vector).type(torch.FloatTensor)
        Y = torch.from_numpy(self.fake_labels[idx]).type(torch.FloatTensor)
        return X, Y
