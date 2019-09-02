import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import sample_1d_dist

""" One-hot encoding not implemented """


class MaskedLinear(nn.Linear):
    """ same as Linear except has a configurable mask on the weights """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))

    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)


class MADE(torch.nn.Module):
    # adapted from https://github.com/karpathy/pytorch-made/blob/master/made.py
    def __init__(self, input_dim_values, hidden_dims, n_masks=1, resample_every=20):
        super().__init__()
        self.n_input_val = input_dim_values
        self.nin = len(input_dim_values)
        self.nout = sum(input_dim_values)
        self.hidden_dims = hidden_dims
        self.num_masks = n_masks
        self.resample_every = resample_every
        self.mask_counter = 0

        hs = [self.nin] + hidden_dims + [self.nout]
        layers = []
        for hin, hout in zip(hs, hs[1:]):
            layers.extend([MaskedLinear(hin, hout),
                           nn.ReLU()])
        layers.pop()
        self.net = nn.Sequential(*layers)
        self.seed = 42

        self.m = {}
        self.update_masks()

    def update_masks(self):
        if self.m and self.num_masks == 1: return  # only a single seed, skip for efficiency
        L = len(self.hidden_dims)

        # fetch the next seed and construct a random stream
        rng = np.random.RandomState(self.seed)
        self.seed = (self.seed + 1) % self.num_masks

        # sample the order of the inputs and the connectivity of all neurons
        self.m[-1] = rng.permutation(self.nin)
        for l in range(L):
            self.m[l] = rng.randint(self.m[l - 1].min(), self.nin - 1, size=self.hidden_dims[l])

        # construct the mask matrices
        masks = [self.m[l - 1][:, None] <= self.m[l][None, :] for l in range(L)]
        masks.append(self.m[L - 1][:, None] < self.m[-1][None, :])

        # handle the case where nout = nin * k, for integer k > 1
        if self.nout > self.nin:
            k = int(self.nout / self.nin)
            # replicate the mask across the other outputs
            masks[-1] = np.repeat(masks[-1].reshape(masks[-1].shape[1], -1, 1), k, axis=2).reshape(-1, self.nout)

        # set the masks in all MaskedLinear layers
        layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]
        for l, m in zip(layers, masks):
            l.set_mask(m)

    def forward(self, x):
        # x has shape (N, d)
        #         sample connectivity
        # mask = self.masks[np.random.randint(len(self.masks))]
        if self.mask_counter == self.resample_every:
            self.update_masks()
            self.mask_counter = 0
        self.mask_counter += 1

        # forward pass through hidden layers
        scores = self.net(x).view(x.shape[0], self.nin, -1)

        # Feed through softmax
        logp = 0
        for j in range(self.nin):
            logp += F.log_softmax(scores[:, j].squeeze(), dim=1)[np.arange(x.shape[0]), x.type(torch.long)[:, j]]

        return logp

    def sample(self, n_samples):
        if not self.m:
            self.update_masks()

        with torch.no_grad():
            samples = torch.zeros(n_samples, self.nin, dtype=torch.float32)
            for n in range(self.num_masks):
                print(self.m[-1])
                # get samples
                curr_samples = torch.zeros(n_samples, self.nin, dtype=torch.float32)
                for j in np.argsort(self.m[-1]):
                    scores = self.net(curr_samples).view(n_samples, self.nin, -1)
                    probs = F.softmax(scores[:, j].squeeze(), dim=1)
                    # inefficient; might have to improve for high-dimensional data
                    for i in range(probs.shape[0]):
                        # loop over sampling chains
                        curr_samples[i, j] = torch.tensor(sample_1d_dist(1, probs[i]))
                # add them to running average
                samples += curr_samples
                self.update_masks()

        return samples.numpy() / self.num_masks
