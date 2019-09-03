import torch
import torch.nn as nn
import torch.nn.functional as F
from util import to_gaussian_params, sample_gaussian


class NetA(nn.Module):
    def __init__(self, n_in=2, n_hidden=100):
        super(NetA, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_in * 2)
        )

    def forward(self, input):
        output = self.net(input)
        return output


class NetB(nn.Module):
    def __init__(self, n_in=2, n_hidden=100):
        super(NetB, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_in + 1)
        )

    def forward(self, input):
        output = self.net(input)
        return output

