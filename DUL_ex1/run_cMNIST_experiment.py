import numpy as np
import torch
from trixi.util import Config
from experiment import coloredMNISTexperiment
from util import plot_dependency_map
import matplotlib.pyplot as plt

c = Config()
c.batch_size = 128
c.n_epochs = 50
c.learning_rate = 0.001
if torch.cuda.is_available():
    c.use_cuda = True
else:
    c.use_cuda = False
c.rnd_seed = 1
c.log_interval = 100


exp = coloredMNISTexperiment(config=c, name='coloredMNIST', n_epochs=c.n_epochs,
                             seed=c.rnd_seed, base_dir='./experiment_dir',
                             loggers={"visdom": ["visdom", {"exp_name": "myenv"}]})

exp.run()

samples = np.transpose(exp.model.sample(9, (28, 28), device=exp.device), (0, 2, 3, 1))

grid_len = 3
fig1, axes = plt.subplots(grid_len, grid_len, figsize=(20, 20))
i = 0
for ax in axes.flat:
    if i >= len(samples):
        break
    ax.imshow(samples[i])
    ax.axis('off')
    i += 1
fig1.tight_layout()
fig1.savefig('mysamples2.png')
