import torch as th
from torch.utils.data import Dataset
import pickle


class CelebLQ(Dataset):
    def __init__(self, path, train=True, transform=None):
        super(CelebLQ, self).__init__()
        with open(path, 'rb') as f:
            data = pickle.load(f)

        if train:
            key = 'train'
        else:
            key = 'test'
        data = data[key]
        self.transform = transform

        # R and B channel of dataset are switched
        self.data = th.tensor(data).index_select(-1, th.arange(2, -1, -1))
        self.data = self.data.permute(0, 3, 1, 2)
        self.data = self.data.float() / 3

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample


class Dequantize(object):
    def __init__(self, max_val):
        self.max_val = max_val

    def __call__(self, img):
        continuous_img = (img * self.max_val + th.rand_like(img)) / (self.max_val + 1)
        return continuous_img
