import torch
from torch.utils.data.dataset import Dataset
import numpy as np

class fake_dataset(Dataset):
    def __init__(self, len_dataset, vector_dim):
        super(fake_dataset, self).__init__()
        self.random_vector = np.random.rand(len_dataset, vector_dim)
        self.random_vector[:, 0]  = self.random_vector[:, 0] * 2 - 1
        self.fake_labels = np.zeros(len_dataset)
    def __len__(self):
        return self.random_vector.shape[0]

    def __getitem__(self, i):
        X = torch.from_numpy(self.random_vector[i])
        Y = torch.from_numpy(self.random_vector[i])
        return X, Y
