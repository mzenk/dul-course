import torch as th
from torch import nn
import torch.nn.functional as F
import util as ut


class CouplingLayer(nn.Module):
    def __init__(self, mask, input_dim=2, hidden_dim=10):
        super().__init__()
        self.s = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            # nn.BatchNorm1d(input_dim),
            nn.Tanh(),
        )
        self.t = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            # nn.BatchNorm1d(input_dim),
        )
        self.register_buffer('mask', mask)
        self.inverse = False

    def forward(self, input):
        if type(input) is tuple:
            input = input[0]
        scale = self.s(self.mask * input)
        transl = self.t(self.mask * input)
        if self.inverse:
            output = self.mask * input + (1 - self.mask) * ut.inv_affine_trf(input, scale, transl)
            log_det = None
        else:
            output = self.mask * input + (1 - self.mask) * ut.affine_trf(input, scale, transl)
            log_det = ((1 - self.mask) * scale).sum(1)
        return output, log_det

    def det_jacobian(self, input):
        """

        :param input: shape (batch, dim)
        :return: determinant of jacobian for x = input; shape (batch,)
        """
        scale = self.s(self.mask * input)
        return th.exp(((1 - self.mask) * scale).sum(1))

    def jac_diag(self, input):
        scale = self.s(self.mask * input)
        return th.exp((1 - self.mask) * scale)


class AffineCoupling(nn.Module):
    # this class adopts the architecture of the exercise 2, HW2 of the DUL course
    def __init__(self, input_channels, n_filters, n_blocks=8, mask_type='checkerboard', reverse_mask=False):
        super().__init__()
        self.net = SimpleResnet(input_channels, n_filters, n_blocks)
        self.n_in = input_channels
        self.rescale = nn.utils.weight_norm(Rescale(input_channels))
        self.reverse_mask = reverse_mask
        self.mask_type = mask_type

    def forward(self, input, inverse=False):
        return self.forward_masked(input)
        x1, x2 = input
        if inverse:
            y1 = x1
            s, t = th.chunk(self.net(x1), 2, dim=1)
            s = self.rescale(th.tanh(s))
            y2 = x2 * th.exp(-s) - t
            log_det = None
        else:
            y1 = x1
            s, t = th.chunk(self.net(x1), 2, dim=1)
            s = self.rescale(th.tanh(s))
            y2 = (x2 + t) * th.exp(s)
            log_det = s.view(s.size(0), -1).sum(1)
        return (y1, y2), log_det

    def forward_masked(self, input, inverse=False):
        # todo for this to work, RealNVP has to be adapted (initialization of coupling layers, flows)
        # get mask
        assert self.mask_type == 'checkerboard'
        mask = ut.get_checkerboard_mask(*input.shape[-2:]).float()
        if self.reverse_mask:
            mask = 1 - mask
        mask = mask.to(input.device)

        s, t = th.chunk(self.net(mask * input), 2, dim=1)
        s = s * (1 - mask)
        t = t * (1 - mask)
        s = self.rescale(th.tanh(s))

        if inverse:
            output = input * th.exp(-s) - t
            log_det = None
        else:
            output = (input + t) * th.exp(s)
            log_det = s.view(s.size(0), -1).sum(-1)
        return output, log_det


class SimpleResnet(nn.Module):
    def __init__(self, n_in, n_filters=256, n_blocks=8):
        super(SimpleResnet, self).__init__()
        n_out = 2 * n_in
        self.n_blocks = n_blocks
        self.conv1 = nn.Conv2d(n_in, n_filters, 3, padding=1)
        self.net = nn.Sequential(
            nn.Conv2d(n_filters, n_filters, 1),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
            nn.Conv2d(n_filters, n_filters, 3, padding=1),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
            nn.Conv2d(n_filters, n_filters, 1),
            nn.BatchNorm2d(n_filters),
        )
        self.convL = nn.Conv2d(n_filters, n_out, 3, padding=1)
        self.bnL = nn.BatchNorm2d(n_out)
        # # just for testing classification... erase afterwards
        # self.fc = nn.Linear(n_filters * 28**2, 10)

    def forward(self, x, classify=False):
        h = self.conv1(x)
        for i in range(self.n_blocks):
            fw_out = self.net(h)
            h = F.relu(fw_out + h)

        output = self.convL(h)
        output = self.bnL(output)
        # # just for testing classification... erase afterwards
        # h = h.flatten(1)
        # output = self.fc(h)
        return output


class SigmoidLayer(nn.Module):
    def __init__(self):
        super(SigmoidLayer, self).__init__()
        self.inverse = False

    def forward(self, input):
        if type(input) is tuple:
            input = input[0]
        if self.inverse:
            output = th.log(input / (1 - input))
            log_det = None
        else:
            output = th.sigmoid(input)
            log_det = (th.log(th.sigmoid(input)) + th.log(1 - th.sigmoid(input))).sum(1)
        return output, log_det

    def det_jacobian(self, input):
        """

        :param input: shape (batch, dim)
        :return: determinant of jacobian for x = input; shape (batch,)
        """
        return (th.sigmoid(input) * (1 - th.sigmoid(input))).prod(1)


class ActNorm(nn.Module):
    """
    We propose an actnorm layer (for activation normalizaton), that performs an affine transformation of the activations
    using a scale and bias parameter per channel, similar to batch normalization. These parameters are
    initialized such that the post-actnorm activations per-channel have zero mean and unit variance given
    an initial minibatch of data. This is a form of data dependent initialization (Salimans and Kingma,
    2016). After initialization, the scale and bias are treated as regular trainable parameters that are
    independent of the data.
    """

    def __init__(self, input_dim):
        super().__init__()
        self.register_parameter('bias', nn.Parameter(th.zeros(input_dim), requires_grad=True))
        self.register_parameter('scale', nn.Parameter(th.ones(input_dim), requires_grad=True))
        self.initialized = False
        self.inverse = False

    def forward(self, input):
        if type(input) is tuple:
            input = input[0]
        if not self.initialized and self.training:
            # initialize
            with th.no_grad():
                self.bias.data = input.mean(0)
                self.scale.data = (input - self.bias).std(0)
                tmp = (input - self.bias) / self.scale
                assert tmp.mean(0).allclose(tmp.new_zeros(input.size(1)), atol=1e-07) and \
                       tmp.std(0).allclose(tmp.new_ones(input.size(1)))
            self.initialized = True
        if self.inverse:
            output = input * self.scale + self.bias
            log_det = None
        else:
            output = (input - self.bias) / self.scale
            log_det = - th.log(th.abs(self.scale)).sum()
        return output, log_det

    def det_jacobian(self, input):
        return self.scale.prod()


class ActNormImg(nn.Module):
    def __init__(self, n_channels):
        """

        :param input_dim: tuple of image dim
        """
        super().__init__()
        self.register_parameter('bias', nn.Parameter(th.zeros(1, n_channels, 1, 1), requires_grad=True))
        self.register_parameter('scale', nn.Parameter(th.ones(1, n_channels, 1, 1), requires_grad=True))
        self.initialized = False

    def forward(self, input, inverse=False):
        # input is split into two parts x1, x2 due to Coupling layer
        # concatenate for easier normalization
        x = th.cat(input, dim=-1)
        x1, x2 = input
        if not self.initialized and self.training:
            # initialize
            with th.no_grad():
                self.bias.data = x.mean(dim=(0, 2, 3), keepdim=True)
                self.scale.data = (x - self.bias).std(dim=(0, 2, 3), keepdim=True)
                tmp = (x - self.bias) / self.scale
                assert tmp.mean((0, 2, 3)).allclose(tmp.new_zeros(x.size(1)), atol=2e-07) and \
                       tmp.std((0, 2, 3)).allclose(tmp.new_ones(x.size(1)))
            self.initialized = True
        if inverse:
            y1 = x1 * self.scale + self.bias
            y2 = x2 * self.scale + self.bias
            log_det = None
        else:
            y1 = (x1 - self.bias) / self.scale
            y2 = (x2 - self.bias) / self.scale
            log_det = - th.log(th.abs(self.scale)).sum().unsqueeze(0)
        return (y1, y2), log_det


class Rescale(nn.Module):
    """Per-channel rescaling. Need a proper `nn.Module` so we can wrap it
    with `torch.nn.utils.weight_norm`.
    Args:
        num_channels (int): Number of channels in the input.
    """
    def __init__(self, num_channels):
        super(Rescale, self).__init__()
        self.weight = nn.Parameter(th.ones(num_channels, 1, 1))

    def forward(self, x):
        x = self.weight * x
        return x
