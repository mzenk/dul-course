import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import softmax, log_softmax, relu
from util import sample_1d_dist


class MLP_autoreg(torch.nn.Module):
    def __init__(self, n_values, hidden_dim, one_hot=False):
        # parameters for softmax p(x1)
        # parameters for MLP approximating p(x2|x1)
        super(MLP_autoreg, self).__init__()
        self.theta1 = nn.Parameter(torch.zeros(n_values[0]))
        self.one_hot = one_hot
        if not one_hot:
            # option 1: real-numbered input
            self.linear1 = nn.Linear(1, hidden_dim)
        else:
            # option 2: one-hot encoding -- not tested
            self.linear1 = nn.Linear(n_values[0], hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, n_values[1])

    def forward(self, x):
        # x has shape (N, 2)
        if self.one_hot:
            # apply one-hot encoding
            n_values_x1 = len(self.theta1)
            x_onehot = torch.zeros(x.shape[0], n_values_x1, device=x.device)
            x_onehot[np.arange(x.shape[0]), x.type(torch.long)[:, 0]] = 1
        # p(x1, x2) = p(x1) * p(x2 | x1) => average neg. logliks of individual terms
        logp_x1 = log_softmax(self.theta1, dim=0)[x.type(torch.long)[:, 0]]

        # tanh is an alternative to ReLu nonlinearity
        if self.one_hot:
            h_relu = relu(self.linear1(x_onehot.type(torch.float)))
        else:
            h_relu = relu(self.linear1(x.type(torch.float)[:, 0].reshape(-1, 1)))
        logp_x2 = log_softmax(self.linear2(h_relu), dim=1)[np.arange(x.shape[0]), x.type(torch.long)[:, 1]]
        return logp_x1 + logp_x2

    def sample(self, n_samples):
        with torch.no_grad():
            # sample x1
            p_x1 = softmax(self.theta1, dim=0)
            x1_samples = torch.tensor(sample_1d_dist(n_samples, p_x1), dtype=torch.float)
            # sample x2|x1
            if self.one_hot:
                x1_onehot = torch.zeros(n_samples, len(self.theta1), device=x1_samples.device)
                x1_onehot[np.arange(n_samples), x1_samples.type(torch.long)] = 1
                h_relu = relu(self.linear1(x1_onehot))
            else:
                h_relu = relu(self.linear1(x1_samples.reshape(-1, 1)))
            p_x2 = softmax(self.linear2(h_relu), dim=1).numpy()
            x2_samples = np.zeros(n_samples)
            for i, x1 in enumerate(x1_samples):
                x2_samples[i] = sample_1d_dist(1, p_x2[i])
        return np.dstack((x1_samples.numpy(), x2_samples))
