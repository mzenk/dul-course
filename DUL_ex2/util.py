import torch
import numpy as np

"""

Components of normalizing flow model:
* methods to compute z = f(x) and x = finv(z) -> erf und erf_inv
* sampling method for gaussian
* nll computation -> analytical determinant
* neural networks (or MADE) for Gaussian mixture parameters


"""


def gaussian_mixture_cdf(x, m, v, p):
    """
    :param x: evaluation points [batch, dim, k_mixture]
    :param m: means [batch, dim, k_mixture]
    :param v: variances [batch, dim, k_mixture]
    :return: gaussian CDF of x [batch]

    evtl. mixture components als extradim.
    """
    cdf = .5 * (1 + torch.erf((x - m) / torch.sqrt(2 * v)))
    return (p * cdf).sum(2)


def inv_gaussian_mixture_cdf(z, m, v):
    """
    UNDER CONSTRUCTION: HAVEN'T FOUND SOLUTION TO INVERT MIXTURE OF GAUSSIAN CDFS
    :param z: [batch, k_mixture]
    :param m: [batch, k_mixture]
    :param v: [batch, k_mixture]
    :return: inverse gaussian CDF of z [batch]
    """

    probs = torch.sqrt(2 * v) * torch.erfinv(2 * z - 1) + m
    return probs.sum(1)


def affine_trf(x, s, t):
    return x * torch.exp(s) + t


def inv_affine_trf(y, s, t):
    return (y - t) * torch.exp(-s)


def checkerboard_split(img_tensor):
    """
    splits a batch of img tensors in a checkerboard fashion (cf. RealNVP paper);
    Not clear to me, what shape the output should have (H/2 or W/2? might have an effect on conv layers)
    :param img_tensor: shape (batch, C, H, W)
    :return: length-2-tuple with tensors of shape (batch, C, H, W//2) each
    """
    N, C, H, W = img_tensor.shape
    assert H % 2 == 0 and W % 2 == 0
    # to ensure that result fits in shape H/2, W/2
    mask = get_checkerboard_mask(H, W)
    x1 = img_tensor[:, :, mask].reshape(N, C, H, W // 2)
    x2 = img_tensor[:, :, ~mask].reshape(N, C, H, W // 2)
    return x1, x2


def inv_checkerboard_split(x1, x2):
    N, C, H, W = x1.shape
    W *= 2
    mask = get_checkerboard_mask(H, W)
    img = x1.new_zeros(N, C, H, W)
    img[:, :, mask] = x1.flatten(-2)
    img[:, :, ~mask] = x2.flatten(-2)
    return img


def get_checkerboard_mask(H, W):
    odd_w = torch.arange(W) % 2
    even_w = 1 - odd_w
    odd_h = torch.arange(H) % 2
    even_h = 1 - odd_h
    mask = odd_h.view(-1, 1).mm(odd_w.view(1, -1)) + even_h.view(-1, 1).mm(even_w.view(1, -1))
    return mask.bool()


def get_vertical_split_mask(H, W):
    mask = torch.zeros(H, W)
    mask[:H//2] = 1
    return mask.bool()


def get_horizontal_split_mask(H, W):
    mask = torch.zeros(H, W)
    mask[:, :W//2] = 1
    return mask.bool()


def tuple_flip(tup):
    return tup[1], tup[0]
