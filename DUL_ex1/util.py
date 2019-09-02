import numpy as np
import matplotlib.pyplot as plt


def sample_1d_dist(n_samples, pdf):
    # discrete pdf!
    # draw some samples
    cdf = np.cumsum(pdf)
    samples = np.zeros(n_samples)
    for i in range(n_samples):
        samples[i] = np.nonzero(np.random.rand() < cdf)[0][0]
    return samples


# def sample_1d_dists_torch(n_samples, pdfs):
#     # expects a tensor of shape (n_pdfs, discrete_channels) for pdfs


def sample_2d_dist(n_samples, pdf):
    # option a) rejection sampling
    # ...
    # option b) sample x1 from marginal distrib. using CDF. Then sample from x2 | x1
    marginal_x1 = pdf.sum(axis=1, keepdims=True)
    conditional_x2 = pdf / marginal_x1
    cdf_x1 = np.cumsum(marginal_x1)
    cdf_condx2 = np.cumsum(conditional_x2, axis=1)
    samples = np.zeros((n_samples, 2), dtype=int)
    for i in range(n_samples):
        curr_x1 = np.nonzero(np.random.rand() < cdf_x1)[0][0]
        curr_x2 = np.nonzero(np.random.rand() < cdf_condx2[curr_x1])[0][0]
        samples[i] = [curr_x1, curr_x2]
    return samples


def plot_dependency_map(dep_list, indices):
    fig, axes = plt.subplots(max(1, len(indices)//3), 3, figsize=(20,20))
    channels, height, width = dep_list[0].shape
    i = 0
    for ax in axes.flat:
        if i >= len(indices):
            break
        x, y = indices[i]
        img = np.zeros((height, width, 3), dtype=float)
        img[..., 0] = dep_list[x*height + y].squeeze()
        img[x, y, 1] = 1
        img /= img.max()
        # print(dep_list[x*height + y].squeeze())
        ax.imshow(img)
        ax.axis('off')
        i += 1
    fig.tight_layout()
