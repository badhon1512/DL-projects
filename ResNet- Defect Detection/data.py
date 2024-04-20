from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):

    def __init__(self, data, mode):
        self.data = data
        self.mode = mode

        self._transform = tv.transforms.Compose([
            tv.transforms.ToPILImage(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=train_mean, std=train_std)
        ])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path = self.data.iloc[index, 0]
        image = imread(path)
        image = gray2rgb(image)
        image = self._transform(image).float()

        return image, torch.tensor(self.data.iloc[index, 1:]).float()