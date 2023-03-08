import torch.utils.data as data
import numpy as np
import os
from lib.utils import data_utils
from lib.config import cfg
from torchvision import transforms as T
import imageio
import json
import cv2


class Dataset(data.Dataset):
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        data_root, split, scene = kwargs['data_root'], kwargs['split'], cfg.scene
        self.data = np.load(os.path.join(data_root, scene + '.npy'))
        self.batch_size = 1024

    def __getitem__(self, index):
        x_1, x_2 = self.data[:, :1], self.data[:, 1:32]
        y_1, y_2 = self.data[:, 32:32+128], self.data[:, 32+128:]
        return x_1, x_2, y_1, y_2

    def __len__(self):
        return len(self.data)
