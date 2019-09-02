import numpy as np
import torch
from trixi.util import Config
from experiment import MNISTexperiment
from util import plot_dependency_map
import matplotlib.pyplot as plt

c = Config()
c.batch_size = 128
c.n_epochs = 10
c.learning_rate = 0.001
if torch.cuda.is_available():
    c.use_cuda = True
else:
    c.use_cuda = False
c.rnd_seed = 1
c.log_interval = 100


exp = MNISTexperiment(config=c, name='test', n_epochs=c.n_epochs,
                      seed=c.rnd_seed, base_dir='./experiment_dir',
                      loggers={"visdom": ["visdom", {"exp_name": "myenv"}]})


# # run backpropagation for each dimension to compute what other
# # dimensions it depends on.
# exp.setup()
# d = 28
# x = (np.random.rand(1, 1, d, d) > 0.5).astype(np.float32)
# res = []
# exp.model.eval()
# for i in range(d):
#     for j in range(d):
#         xtr = torch.tensor(x, requires_grad=True, device='cuda:0')
#         xtrhat = exp.model(xtr).squeeze()
#         loss = xtrhat[0, i, j]
#         loss.backward()
#
#         depends = (xtr.grad[0].cpu().numpy() != 0).astype(np.uint8)
#         # print('i, j = {}, {}; max dep.: {}'.format(i, j, np.transpose(np.nonzero(depends[0]))[-1]))
#         res.append(depends)
#
# # plot the dependency for a couple of pixels
# plot_dependency_map(res, [(0,0), (0,12), (0,25),
#                           (12,0), (12,12), (12,25),
#                           (25,0), (25,12), (25,25)])
# plt.show()

exp.run()

samples = exp.model.sample(9, (28, 28), device=exp.device)

grid_len = 3
fig1, axes = plt.subplots(grid_len, grid_len, figsize=(20, 20))
i = 0
for ax in axes.flat:
    if i >= len(samples):
        break
    ax.imshow(samples[i].squeeze(), cmap='gray')
    ax.axis('off')
    i += 1
fig1.tight_layout()
fig1.savefig('mysamples.png')
