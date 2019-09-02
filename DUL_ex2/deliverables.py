# Deliverables (mostly plotting) for both problems
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.utils import make_grid


def plot_nll(nll_train, nll_valid):
    # takes a sequence of training and validation losses and plots them in bits per dimension
    # done by visdom
    pass


def show_density_2d(model, xmin=-4, xmax=4, ymin=-4, ymax=4, gridpoints=100):
    n = gridpoints
    gridx, gridy = np.meshgrid(np.linspace(xmin, xmax, n), np.linspace(ymin, ymax, n))
    flat_grid = np.dstack((gridx, gridy)).reshape(-1, 2)
    _, log_px = model(torch.tensor(flat_grid, dtype=torch.float32))
    pxy = torch.exp(log_px.reshape((n, n))).detach().numpy()
    plt.figure()
    plt.imshow(pxy[::-1], cmap='gray', extent=(xmin, xmax, ymin, ymax))
    plt.colorbar()
    plt.show()
    return pxy


def display_samples(model, n_samples, device='cpu'):
    samples = model.sample(n_samples, device=device).numpy()
    sns.jointplot(samples[:, 0], samples[:, 1])
    plt.show()
    return samples


def display_img_samples(model, n_samples, device='cpu'):
    samples = model.sample(n_samples, device=device)
    img_grid = make_grid(samples).permute((1, 2, 0))
    plt.imshow(img_grid)
    plt.show()
    return samples


def display_latent_vars(latent_rep, targets):
    z1, z2 = latent_rep[:, 0], latent_rep[:, 1]
    n_targets = len(np.unique(targets))
    plt.figure()
    for i in range(n_targets):
        mask = torch.tensor(targets == i, dtype=torch.uint8)
        plt.scatter(z1[mask], z2[mask], c='C' + str(i))
    plt.show()


def grid_transform(model):
    # optional but cool: visualize f by grid distortion
    pass


