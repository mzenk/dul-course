import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedConv2d(nn.Conv2d):
    """ same as Linear except has a configurable mask on the weights """

    def __init__(self, mask_type, *args, **kwargs):
        # kwargs: stride=1, padding=0, groups=1, bias=True
        super().__init__(*args, **kwargs)
        # could optimize memory consumption with broadcasting
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


class ResBlock(nn.Module):
    def __init__(self, h, mask='B'):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            MaskedConv2d(mask, 2 * h, h, 1, ),
            nn.BatchNorm2d(h),
            nn.ReLU(),
            MaskedConv2d(mask, h, h, 3, padding=1),
            nn.BatchNorm2d(h),
            nn.ReLU(),
            MaskedConv2d(mask, h, 2 * h, 1),
            nn.BatchNorm2d(2 * h),
        )

    def forward(self, x):
        return self.net(x) + x


class PixelCNN(nn.Module):
    def __init__(self, img_channels=3, h=128, discrete_channels=4):
        super().__init__()
        self.h = h
        self.img_channels = img_channels
        self.discrete_channels = discrete_channels

        if img_channels == 1:
            mask_modifier = '0'
        else:
            mask_modifier = ''

        layers = [MaskedConv2d('A' + mask_modifier, img_channels, 2 * self.h, 7, padding=3),
                  nn.BatchNorm2d(2 * self.h)
                  ]
        for i in range(12):
            layers.append(ResBlock(self.h, mask='B' + mask_modifier))

        # output layers
        layers.extend([nn.ReLU(),
                       MaskedConv2d('B' + mask_modifier, 2*self.h, self.h, 1),
                       nn.BatchNorm2d(self.h),
                       nn.ReLU(),
                       MaskedConv2d('B' + mask_modifier, self.h, self.img_channels * self.discrete_channels, 1),
                       # nn.BatchNorm2d(self.img_channels * self.discrete_channels),
                       # Batchnorm at last layer raises cuDNN error: CUDNN_STATUS_EXECUTION_FAILED
                       ],)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # returns (N, d_c, C, H, W)
        logits = self.net(x)
        N, C, H, W = logits.shape
        pixelwise_logp = F.log_softmax(logits.view(N, self.img_channels, self.discrete_channels, H, W), dim=2)
        return torch.transpose(pixelwise_logp, 1, 2)

    def sample(self, num_samples, img_shape, device='cuda:0'):
        # samples pixels sequentially
        self.eval()
        samples = torch.zeros(num_samples, self.img_channels, img_shape[0], img_shape[1],
                              dtype=torch.float32, device=device)
        with torch.no_grad():
            for i in range(img_shape[0]):
                for j in range(img_shape[1]):
                    for c in range(self.img_channels):
                        pixelwise_logp = self(samples)
                        pmf = torch.exp(pixelwise_logp[:, :, c, i, j])
                        samples[:, c, i, j] = torch.multinomial(pmf, 1).squeeze().float() / (self.discrete_channels - 1)
        return samples.cpu().numpy()


class SimplePixelCNN(PixelCNN):
    def __init__(self, hidden_dims, img_channels=3, discrete_channels=4):
        super().__init__(img_channels=img_channels, discrete_channels=discrete_channels)
        if img_channels == 1:
            mask_modifier = '0'
        else:
            mask_modifier = ''

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
                       MaskedConv2d('B' + mask_modifier, hidden_dims[-1], img_channels * discrete_channels, 1)
                       ])

        self.net = nn.Sequential(*layers)
