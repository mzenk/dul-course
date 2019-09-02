import numpy as np
import torch
import torch.nn.functional as F


def to_gaussian_params(x, scalar_var=False):
    """
    converts real valued network outputs to valid mean vectors and variances
    :param x: tensor of shape (batch, n_dim * 2) or (batch, n_dim + 1)
    :param scalar_var: in a special case, only a scalar variance value is included
    :return: two tensors tensor of shape (batch,)
    """
    if scalar_var:
        mu, var = torch.split(x, [x.shape[1] - 1, 1], dim=1)
    mu, var = torch.split(x, x.shape[1]//2, dim=1)
    # make variances positive
    var = F.softplus(var) + 1e-8
    return mu, var


def sample_gaussian(mu, var):
    """
    draws a sample from the gaussian distr. specified by mu, std using the reparametrization trick.
    :param mu: mean of shape (batch, n_dim)
    :param var: enries of diagonal variance matrix; shape (batch, n_dim) or (batch,) if same variance for all
    :return:
    """
    if len(var.shape) == 1:
        var.unsqueeze(1)
    return torch.randn_like(mu) * var.sqrt() + mu


def log_normal(x, mu, var):
    """
    computes log prob of gaussian with diagonal of covariance matrix given by var
    :param x: (batch, dim)
    :param mu: (batch, dim)
    :param var: (batch, dim)
    :return: log_prob (batch,)
    """
    element_wise = .5 * (-(x - mu)**2 / var - torch.log(2 * np.pi * var))
    return element_wise.sum(-1)


def kl_normal(q_mu, q_var, p_mu, p_var):
    """

    :param q_mu: mean of q distr. (batch, dim)
    :param q_std: std of q distr. (batch, dim)
    :param p_mu: mean of p distr. (batch, dim)
    :param p_std: std of p distr. (batch, dim)
    :return: kl divergence between distributions (batch,)
    """
    kl = .5 * (torch.log(p_var) - torch.log(q_var) + q_var / p_var + (q_mu - p_mu)**2 / p_var - 1)
    return kl.sum(1)
