import os
import time
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils import data
from torchvision import datasets, transforms, utils
from util import plot_dependency_map
import matplotlib.pyplot as plt
from PIL import Image



# class MaskedConv2d(nn.Conv2d):
#     def __init__(self, mask_type, *args, **kwargs):
#         super(MaskedConv2d, self).__init__(*args, **kwargs)
#         assert mask_type in {'A', 'B'}
#         self.register_buffer('mask', self.weight.data.clone())
#         _, _, kH, kW = self.weight.size()
#         self.mask.fill_(1)
#         self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
#         self.mask[:, :, kH // 2 + 1:] = 0
#
#     def forward(self, x):
#         self.weight.data *= self.mask
#         return super(MaskedConv2d, self).forward(x)

class MaskedConv2d(nn.Conv2d):
    """ same as Linear except has a configurable mask on the weights """

    def __init__(self, mask_type, *args, **kwargs):
        # kwargs: stride=1, padding=0, groups=1, bias=True
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())
        self.mask.fill_(1)
        cout, cin, kH, kW = self.weight.size()
        # split channels to RGB (cf. section 3.4)
        in_rg = cin // 3
        in_b = cin - 2*cin//3
        out_rg = cout // 3
        out_b = cout - 2*cout//3

        self.mask[:, :, kH//2 + 1:] = 0
        self.mask[:, :, kH//2, kW//2:] = 0

        if mask_type == 'A':
            # R -> G
            self.mask[out_rg: 2*out_rg, : in_rg, kH//2, kW//2] = 1
            # R,G -> B
            self.mask[-out_b:, :-in_b, kH // 2, kW // 2] = 1
        elif mask_type == 'B':
            # R -> R
            self.mask[: out_rg, : in_rg, kH // 2, kW // 2] = 1
            # R, G -> G
            self.mask[out_rg: 2*out_rg, : 2*in_rg, kH // 2, kW // 2] = 1
            # R, G, B -> B
            self.mask[-out_b:, :, kH // 2, kW // 2] = 1
        elif mask_type == 'B0':
            # for one-channel image
            if kH == 1:
                # 1x1 conv.
                self.mask[:, :] = 1
            else:
                self.mask[:, :, kH//2, kW//2] = 1

    def forward(self, x):
        return F.conv2d(x, self.mask * self.weight, bias=self.bias, padding=self.padding)


class genMNIST(datasets.MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.targets = self.data[:, None, :, :].long()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def draw_samples(model, input):
    model.eval()
    with torch.no_grad():
        pixelwise_logp = model(input)
        pmf = torch.exp(pixelwise_logp[:, :, i, j])
        samples = torch.multinomial(pmf, 1).float() / 255.
    return samples


mask_modifier = '0'

hidden_dims = [64] * 7
img_channels = 1
discrete_channels = 256
layers = []
for i, h in enumerate(hidden_dims):
    if i == 0:
        layers.extend([MaskedConv2d('A' + mask_modifier, img_channels, hidden_dims[i], 7, padding=3, bias=False),
                       nn.BatchNorm2d(hidden_dims[i])
                       ])
    else:
        layers.extend([nn.ReLU(),
                       MaskedConv2d('B' + mask_modifier, hidden_dims[i - 1], hidden_dims[i], 3, padding=1, bias=False),
                       nn.BatchNorm2d(hidden_dims[i])
                       ])

# output layers
layers.extend([nn.ReLU(),
               MaskedConv2d('B' + mask_modifier, hidden_dims[-1], img_channels * discrete_channels, 1),
               nn.LogSoftmax(dim=1)
               ])

net = nn.Sequential(*layers)

# fm = 64
# net = nn.Sequential(
#         MaskedConv2d('A' + mask_modifier, 1,  fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
#         MaskedConv2d('B' + mask_modifier, fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
#         MaskedConv2d('B' + mask_modifier, fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
#         MaskedConv2d('B' + mask_modifier, fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
#         MaskedConv2d('B' + mask_modifier, fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
#         MaskedConv2d('B' + mask_modifier, fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
#         MaskedConv2d('B' + mask_modifier, fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
#         MaskedConv2d('B' + mask_modifier, fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
#         nn.Conv2d(fm, 256, 1),
#         nn.LogSoftmax(dim=1)
#     )

print(net)
net.to('cuda:0')

dataset_train = genMNIST('.', train=True, download=True, transform=transforms.ToTensor())
dataset_test = genMNIST('.', train=False, download=True, transform=transforms.ToTensor())

data_loader_kwargs = {'num_workers': 1, 'pin_memory': True}
train_data_loader = torch.utils.data.DataLoader(dataset_train, batch_size=128, shuffle=True, **data_loader_kwargs)
test_data_loader = torch.utils.data.DataLoader(dataset_test, batch_size=128, shuffle=True, **data_loader_kwargs)

# tr = data.DataLoader(datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor()),
#                      batch_size=128, shuffle=True, num_workers=1, pin_memory=True)
# te = data.DataLoader(datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor()),
#                      batch_size=128, shuffle=False, num_workers=1, pin_memory=True)

sample = torch.Tensor(144, 1, 28, 28).cuda()
optimizer = optim.Adam(net.parameters())
for epoch in range(25):
    # train
    err_tr = []
    time_tr = time.time()
    net.train(True)
    # for input, _ in tr:
    #     input = input.cuda()
    #     target = (input.data[:, 0] * 255).long()
    for input, target in train_data_loader:
        input = input.cuda()
        target = target.squeeze().cuda()
        # loss = F.cross_entropy(net(input), target)
        loss = F.nll_loss(net(input), target, reduction='none').sum() / input.shape[0]
        err_tr.append(loss.item() / np.log(2) / 28**2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    time_tr = time.time() - time_tr

    # compute error on test set
    err_te = []
    time_te = time.time()
    net.train(False)
    # for input, _ in te:
    #     input = input.cuda()
    #     target = (input.data[:, 0] * 255).long()
    for input, target in test_data_loader:
        input = input.cuda()
        target = target.squeeze().cuda()
        loss = F.cross_entropy(net(input), target)
        err_te.append(loss.item())
    time_te = time.time() - time_te

    # sample
    if epoch % 3 == 0:
        sample.fill_(0)
        net.train(False)
        for i in range(28):
            for j in range(28):
                # out = net(sample)
                # probs = torch.exp(out[:, :, i, j])
                # sample[:, :, i, j] = torch.multinomial(probs, 1).float() / 255.
                sample[:, :, i, j] = draw_samples(net, sample)
        utils.save_image(sample, 'sample_{:02d}.png'.format(epoch), nrow=12, padding=0)
        plt.figure()
        plt.hist(sample.flatten().cpu().numpy(), bins=255, range=(0.1, 1))
        # ignores zero pixels
        plt.savefig('samplehist{:02d}.png'.format(epoch))

    print('epoch={}; nll_tr={:.7f}; nll_te={:.7f}; time_tr={:.1f}s; time_te={:.1f}s'.format(
        epoch, np.mean(err_tr), np.mean(err_te), time_tr, time_te))
