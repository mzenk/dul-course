import torch as th
from torch import nn, optim, distributions
import numpy as np
import util as ut
from layers import CouplingLayer, SigmoidLayer, ActNorm, AffineCoupling, ActNormImg


class ARFlow2D(nn.Module):
    def __init__(self, input_dim=2, k_mixture=3):
        self.k_mixture = k_mixture
        # replace these with networks
        self.prior_mixture = th.empty(2, k_mixture).fill_(1 / k_mixture)
        self.means = th.rand(input_dim, k_mixture)
        self.vars = th.abs(th.rand(input_dim, k_mixture))

    def forward(self, *input):
        pass

    def flow(self, x):
        pass

    def inv_flow(self, z):
        pass

    def compute_nll(self, x):
        """
        The first term of the LL is constant because of the uniform density and can be left out
        :param x:
        :return:
        """

        # both are   mixture of gaussians
        detf11 = 0
        detf22 = 0
        nll = - (detf11 + detf22)
        return nll


class RealNVP2D(nn.Module):
    def __init__(self, input_dim=2, n_coupling=4, hidden_dim=10, prior='uniform'):
        super(RealNVP2D, self).__init__()
        self.n_coupling = n_coupling
        self.input_dim = input_dim
        layers = []
        for i in range(n_coupling):
            mask = th.zeros(2)
            mask[i % 2] = 1.
            # layers.append(CouplingLayer(mask, input_dim=input_dim))
            layers.extend([CouplingLayer(mask, input_dim=input_dim, hidden_dim=hidden_dim),
                           ActNorm(input_dim)])
        # layers.pop()
        if prior == 'gauss':

            # self.prior_logprob = lambda x: distributions.multivariate_normal.MultivariateNormal(
            #     th.zeros(x.shape[1]), th.eye(x.shape[1])).log_prob(x)
            self.prior = distributions.multivariate_normal.MultivariateNormal(th.zeros(input_dim), th.eye(input_dim))
        else:
            # assume uniform prior
            layers.append(SigmoidLayer())
            self.prior = distributions.uniform.Uniform(th.zeros(input_dim), th.ones(input_dim))
        self.fw_net = nn.Sequential(*layers)
        self.bw_net = nn.Sequential(*layers[::-1])

    def forward(self, input):
        """
        Computes the log likelihood of the input. Because of the uniform target distribution, the first term in eq. (5) vanishes.
        :param input: data of shape (batch, dim)
        :return: log likelihood of data of shape (batch,)
        """
        # Important: log likelihood involves determinant. p(f(x)) term is zero because of uniform distribution
        log_det = 0
        output = input
        for m in self.fw_net.children():
            output, ld = m(output)
            log_det += ld

        prior_logp = self.prior.log_prob(output.cpu()).to(output.device)
        if type(self.prior) == distributions.uniform.Uniform:
            prior_logp = prior_logp[:, 0].squeeze()
        log_det += prior_logp
        return output, log_det

    def flow(self, x):
        return self.fw_net(x)[0]

    def inv_flow(self, z):
        for m in self.bw_net.children():
            if type(m) in (CouplingLayer, SigmoidLayer, ActNorm):
                m.inverse = True

        output, _ = self.bw_net(z)
        # restore normal functionality
        for m in self.bw_net.children():
            if type(m) in (CouplingLayer, SigmoidLayer, ActNorm):
                m.inverse = False

        return output

    def sample(self, num_samples, device='cuda:0'):
        with th.no_grad():

            # generate samples from z distribution
            z = self.prior.sample((num_samples,)).to(device)
            # apply inverse flow to get samples for x
            return self.inv_flow(z)


class RealNVP(nn.Module):
    def __init__(self, img_dim, n_coupling=4, n_filters=128):
        """

        :param img_dim: tensor style (C, H, W)
        :param n_coupling:
        :param n_filters:
        """
        super(RealNVP, self).__init__()
        self.n_coupling = n_coupling
        self.img_dim = img_dim
        self.layers = []
        for i in range(n_coupling):
            name = 'coupling' + str(i)
            # self.add_module(name, AffineCoupling(img_dim[0], n_filters, n_coupling))
            self.add_module(name, AffineCoupling(img_dim[0], n_filters, n_coupling, reverse_mask=i % 2))
            self.layers.append(name)

            # name = 'actnorm' + str(i)
            # self.add_module(name, ActNormImg(img_dim[0]))
            # self.layers.append(name)

    def forward(self, input):
        """
        Computes the log likelihood of the input. Because of the uniform target distribution, the first term in eq. (5) vanishes.
        :param input: data of shape (batch, dim)
        :return: log likelihood of data of shape (batch,)
        """
        # x, preprocess_logdet = self._preprocess(input)
        x, preprocess_logdet = input, 0
        # Important: log likelihood involves determinant.
        z, log_det = self.flow_masked(x, accumulate_det=True)
        log_det += preprocess_logdet

        # Gaussian prior
        prior_logp = -.5 * th.sum(z.view(z.shape[0], -1)**2, dim=1) - .5 * np.prod(z.shape[1:]) * np.log(2 * np.pi)
        return z, log_det + prior_logp

    def _preprocess(self, x):
        # dequantization is done on dataset side
        # transform to logits (with constant alpha to avoid divergence)
        alpha = 1e-5
        y = .5 * alpha + (1 - alpha) * x
        logits = y.log() - (1 - y).log()
        log_det = np.log(1 - alpha) * (th.log(y) + th.log(1 - y)).view(y.size(0), -1).sum(1)
        return logits, log_det

    def flow(self, x, accumulate_det=False):
        # checkerboard split
        # output = ut.checkerboard_split(x)
        # suboptimal split; just for testing
        output = th.split(x, x.shape[-1]//2, dim=-1)
        log_det = 0
        for i, name in enumerate(self.layers):
            layer = self._modules[name]
            if i > 0:
                output = ut.tuple_flip(output)
            output, ld = layer(output)
            if accumulate_det:
                log_det += ld
        # z = ut.inv_checkerboard_split(*output)
        # suboptimal split; just for testing
        z = th.cat(output, dim=-1)

        if accumulate_det:
            return z, log_det
        return z

    def flow_masked(self, x, accumulate_det=False):
        log_det = 0
        z = x
        for i, name in enumerate(self.layers):
            layer = self._modules[name]
            z, ld = layer(z)
            if accumulate_det:
                log_det += ld
        if accumulate_det:
            return z, log_det
        return z

    def inv_flow(self, z):
        # checkerboard split
        output = ut.checkerboard_split(z)
        # suboptimal split; just for testing
        # output = th.split(z, z.shape[-1]//2, dim=-1)
        for i, name in enumerate(self.layers[::-1]):
            layer = self._modules[name]
            if i > 0:
                output = ut.tuple_flip(output)
            output, _ = layer(output, inverse=True)
        x = ut.inv_checkerboard_split(*output)
        # suboptimal split; just for testing
        # x = th.cat(output, dim=-1)
        return x

    def inv_flow_masked(self, z):
        x = z
        for i, name in enumerate(self.layers[::-1]):
            layer = self._modules[name]
            x, _ = layer(x, inverse=True)
        return x

    def sample(self, num_samples, device='cuda:0'):
        with th.no_grad():

            # generate samples from z distribution
            z = th.randn(num_samples, *self.img_dim).to(device)
            # apply inverse flow to get samples for x
            # return th.sigmoid(self.inv_flow(z))
            return self.inv_flow_masked(z)
