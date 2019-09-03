import torch
import torch.nn as nn
from networks import NetA, NetB
import util as ut


class VAE2D(nn.Module):
    def __init__(self, dec_type=''):
        super(VAE2D, self).__init__()
        self.enc = Encoder2D()
        if dec_type == 'B':
            self.dec = DecoderB2D()
        else:
            self.dec = DecoderA2D()
        self.register_buffer('prior_mu', torch.zeros(2))
        self.register_buffer('prior_var', torch.ones(2))

    def forward(self, input):
        return self.compute_nelbo(input)

    def compute_nelbo(self, x):
        # encode x and draw sample
        mu_z, var_z = self.enc(x)
        z = ut.sample_gaussian(mu_z, var_z)
        mu_x, var_x = self.dec(z)

        dkl = ut.kl_normal(mu_z, var_z, self.prior_mu, self.prior_var)
        recon = - ut.log_normal(x, mu_x, var_x)
        nelbo = dkl + recon
        return nelbo.mean(), dkl.mean(), recon.mean()

    def sample_z_from_x(self, x):
        return ut.sample_gaussian(*self.enc(x))

    def sample_z(self, num_samples):
        return ut.sample_gaussian(self.prior_mu.expand(num_samples, -1), self.prior_var.expand(num_samples, -1))

    def sample_x(self, num_samples):
        return self.sample_x_from_z(self.sample_z(num_samples))

    def sample_x_from_z(self, z):
        mu, var = self.dec(z)
        return ut.sample_gaussian(mu, var), mu


class Encoder2D(nn.Module):
    def __init__(self):
        super(Encoder2D, self).__init__()
        self.net = NetA()

    def forward(self, input):
        output = self.net(input)
        mu, var = ut.to_gaussian_params(output)
        return mu, var


class DecoderA2D(nn.Module):
    def __init__(self):
        super(DecoderA2D, self).__init__()
        self.net = NetA()

    def forward(self, input):
        output = self.net(input)
        mu, var = ut.to_gaussian_params(output)
        return mu, var


class DecoderB2D(nn.Module):
    def __init__(self):
        super(DecoderB2D, self).__init__()
        self.net = NetB()

    def forward(self, input):
        output = self.net(input)
        mu, var = ut.to_gaussian_params(output, scalar_var=True)
        return mu, var


if __name__ == '__main__':
    enc = Encoder2D()
    decA = DecoderA2D()
    decB = DecoderB2D()

    input = torch.rand(7, 2)

    # m, v = enc(input)
    # z = ut.sample_gaussian(m, v)
    # output = decA(z)
    # print(m, v, z)
    #
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    #
    # n_samples = 10000
    # m = 2. * torch.ones(n_samples, 2)
    # v = .5 * torch.ones(n_samples, 2)
    # z = ut.sample_gaussian(m, v)
    #
    # sns.jointplot(z[:, 0], z[:, 1])
    # plt.show()

    vae = VAEtypeA()
    loss = vae(input)
    print(loss[0].shape)

