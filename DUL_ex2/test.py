import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from models import RealNVP2D, RealNVP
import torch
from torch.utils.data import DataLoader
from experiment import SmileyExperiment, MNISTExperiment, CelebAExperiment, MNIST_classification
from trixi.util import Config
from deliverables import show_density_2d, display_latent_vars
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from datasets import Dequantize


def sample_data():
    count = 100000
    rand = np.random.RandomState(0)
    a = [[-1.5, 2.5]] + rand.randn(count // 3, 2) * 0.2
    b = [[1.5, 2.5]] + rand.randn(count // 3, 2) * 0.2
    c = np.c_[2 * np.cos(np.linspace(0, np.pi, count // 3)),
              -np.sin(np.linspace(0, np.pi, count // 3))]
    c += rand.randn(*c.shape) * 0.2
    data_x = np.concatenate([a, b, c], axis=0)
    data_y = np.array([0] * len(a) + [1] * len(b) + [2] * len(c))
    perm = rand.permutation(len(data_x))
    return data_x[perm], data_y[perm]


def numerical_deriv(input_dim, func, x=None, n_avg=100, delta=1e-4):
    # returns shape (n_avg,)
    if x is None:
        x = torch.rand(n_avg, input_dim)
    x = x.double()

    try:
        func.double()
    except AttributeError:
        pass
    # this computes the derivative of xi wrt xi (which should be just what we need for the realNVP case
    deriv = x.new_empty(x.size())
    for i in range(x.shape[1]):
        x1 = x - .5 * delta * torch.tensor([i == 0, i == 1], dtype=torch.double)
        x2 = x + .5 * delta * torch.tensor([i == 0, i == 1], dtype=torch.double)
        deriv[:, i] = (func(x2) - func(x1))[:, i] / delta
    try:
        func.float()
    except AttributeError:
        pass
    return deriv


def numerical_det(diagonal_jac):
    # returns shape (n_avg,)
    return diagonal_jac.prod(1)


def square(x):
    return x ** 2


def log_gauss(x):
    return - .5 * (x * x).sum(1) / 1 ** 2


def test_flow2D():
    # ==== Basic functionality tests ====
    torch.manual_seed(42)
    test_data, test_targets = sample_data()

    device = 'cpu'

    model = RealNVP2D(n_coupling=3, prior='uniform')

    d = torch.tensor(test_data, dtype=torch.float32)[:5]

    # test flows
    z = model.flow(d)
    print(z)

    recon = model.inv_flow(z)
    print(recon)
    print(recon.isclose(d))

    model(d)
    samples = model.sample(1000, device=device).numpy()
    sns.jointplot(samples[:, 0], samples[:, 1])
    plt.show()

    d = torch.tensor(test_data, dtype=torch.float32)
    # display_latent_vars(test_data, test_targets)
    show_density_2d(model)
    with torch.no_grad():
        z = model(d)[0].detach()
    display_latent_vars(z, test_targets)


# ==== Jacobian tests ====
def test_jacobian():
    model = RealNVP2D(n_coupling=3, prior='uniform')
    # x = d
    # x = torch.rand(5, 2)
    x = torch.ones(5, 2) * np.log(2)
    print(x)
    # test determinants
    print('=====')
    print('Determinant check')
    for m in model.fw_net.children():
        print(type(m))
        with torch.no_grad():
            det1 = m.det_jacobian(x)
        det2 = numerical_det(numerical_deriv(2, m, x=x))

        mean_rel_error = (torch.abs(det1.double() - det2)).mean()
        print(mean_rel_error)
        if mean_rel_error > 1e-3:
            print(det1)
            print(det2)
        break

    print('=====')
    print('Derivative check')
    with torch.no_grad():
        for m in model.fw_net.children():
            print(type(m))
            try:
                deriv1 = m.jac_diag(x)
            except AttributeError:
                continue
            deriv2 = numerical_deriv(2, m, x=x)

            mean_rel_error = (torch.abs(deriv1.double() - deriv2)).mean()
            print(mean_rel_error)
            # if mean_rel_error > 1e-3:
            print(deriv1)
            print(deriv2)
            break

    print('=====')
    print('sanity check')
    det1 = torch.exp(d).prod(1)
    det2 = numerical_det(numerical_deriv(2, torch.exp, x=d))
    print((torch.abs(det1.double() - det2) / torch.abs(det2)).mean())
    deriv1 = torch.exp(d)
    deriv2 = numerical_deriv(2, torch.exp, x=d, delta=1e-3)
    print((torch.abs(deriv1.double() - deriv2) / torch.abs(deriv2)).mean())


# ==== Test experiment run ====
def test_2Dexperiment():
    c = Config()

    c.batch_size = 200
    c.n_epochs = 40
    c.learning_rate = 0.001
    if torch.cuda.is_available():
        c.use_cuda = True
    else:
        c.use_cuda = False
    c.rnd_seed = 1
    c.log_interval = 200
    # model-specific
    c.n_coupling = 8
    c.prior = 'gauss'

    exp = SmileyExperiment(c, name='gauss', n_epochs=c.n_epochs, seed=42, base_dir='experiment_dir',
                           loggers={'visdom': ['visdom', {"exp_name": "myenv"}]})

    exp.run()

    # sampling
    samples = exp.model.sample(1000).cpu().numpy()
    sns.jointplot(samples[:, 0], samples[:, 1])
    plt.show()


def test_Celeb_experiment():
    c = Config()

    c.batch_size = 64
    c.n_epochs = 15
    c.learning_rate = 0.001
    if torch.cuda.is_available():
        c.use_cuda = True
    else:
        c.use_cuda = False
    c.rnd_seed = 1
    c.log_interval = 100
    # model-specific
    c.n_coupling = 8
    c.n_filters = 64

    exp = CelebAExperiment(c, name='test', n_epochs=c.n_epochs, seed=42, base_dir='experiment_dir',
                           loggers={'visdom': ['visdom', {"exp_name": "myenv"}]})

    exp.run()

    exp.model.eval()
    exp.model.to('cpu')
    with torch.no_grad():
        samples = exp.model.sample(16, device='cpu')
        img_grid = make_grid(samples).permute((1, 2, 0))
    plt.imshow(img_grid)
    plt.show()
    return exp.model


def test_MNIST_experiment():
    c = Config()

    c.batch_size = 64
    c.n_epochs = 50
    c.learning_rate = 0.001
    c.weight_decay = 5e-5
    if torch.cuda.is_available():
        c.use_cuda = True
    else:
        c.use_cuda = False
    c.rnd_seed = 1
    c.log_interval = 100
    c.subset_size = 10
    # model-specific
    c.n_coupling = 8
    c.n_filters = 64

    exp = MNISTExperiment(c, name='mnist_test', n_epochs=c.n_epochs, seed=42, base_dir='experiment_dir',
                           loggers={'visdom': ['visdom', {"exp_name": "myenv"}]})

    exp.run()

    exp.model.eval()
    exp.model.to('cpu')
    with torch.no_grad():
        samples = exp.model.sample(16, device='cpu')
        img_grid = make_grid(samples).permute((1, 2, 0))
    plt.imshow(img_grid)
    plt.show()
    return exp.model


# ==== ResNet test ====
def test_Resnet():
    c = Config()

    c.batch_size = 64
    c.batch_size_test = 1000
    c.n_epochs = 10
    c.learning_rate = 0.01
    c.momentum = 0.9
    if torch.cuda.is_available():
        c.use_cuda = True
    else:
        c.use_cuda = False
    c.rnd_seed = 1
    c.log_interval = 200

    exp = MNIST_classification(config=c, name='experiment', n_epochs=c.n_epochs,
                           seed=42, base_dir='./experiment_dir', loggers={"visdom": "visdom"})

    exp.run()


def test_flow():
    # ==== Basic functionality tests ====
    transform = transforms.Compose(
        [transforms.ToTensor(),
        Dequantize(255)]
    )
    dataset = datasets.MNIST(root="data/", download=True, transform=transform, train=False)
    dataloader = DataLoader(dataset, batch_size=5)

    device = 'cpu'

    model = RealNVP((1, 28, 28), n_coupling=1, n_filters=100)

    d, target = next(iter(dataloader))

    # test flows
    with torch.no_grad():
        z, ld = model(d)
        recon = torch.sigmoid(model.inv_flow(z))
        abs_error = torch.abs(recon - d)
        rel_error = torch.abs(recon - d) / (torch.abs(d) + 1e-8)
    print(recon.allclose(d, atol=1e-6))
    print('Largest absolute error:')
    print(abs_error.max())
    print('between {} and reconstruction {}'.format(
        d.flatten()[abs_error.argmax()], recon.flatten()[abs_error.argmax()]))

    print('Largest relative error:')
    print(rel_error.max())
    print('between {} and reconstruction {}'.format(
        d.flatten()[rel_error.argmax()], recon.flatten()[rel_error.argmax()]))
    # for atol = 1e-6, this is true, but the relative error is still large. is this ok?

    return d, z, recon, ld


if __name__ == '__main__':
    torch.manual_seed(98765)
    # d, z, recon, ld = test_flow()

    model = test_MNIST_experiment()
