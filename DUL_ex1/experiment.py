import numpy as np
import torch
from torch.optim import Adam, SGD
import torch.nn.functional as F
import torch.nn as nn
from trixi.experiment import PytorchExperiment
from torchvision import transforms, datasets
import pickle
from pixelCNN import PixelCNN, SimplePixelCNN
from PIL import Image


# todo: if data centering is desired need to find a way without destroying the targets


class ColoredMNIST(torch.utils.data.Dataset):
    def __init__(self, path, train=True, transform=None):
        super().__init__()
        # load dataset; assumed to be .pkl
        with open(path, 'rb') as f:
            if train:
                self.data = pickle.load(f)['train'].astype(np.float32)
            else:
                self.data = pickle.load(f)['test'].astype(np.float32)
        self.transform = transform
        self.targets = torch.tensor(np.transpose(self.data, (0, 3, 1, 2)), dtype=torch.long)
        # center images channel-wise
        # self.data -= self.data.mean(axis=(0, 1, 2))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        if self.transform:
            img = self.transform(img)
        return img, self.targets[idx]


class ToyDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        # load dataset; assumed to be .pkl
        # center images channel-wise
        n_pxls = 10
        data = torch.zeros(2*n_pxls + 2, 1, n_pxls, n_pxls)
        # couple of bars
        for i in range(n_pxls):
            # vertical
            data[i, :, i, :] = 1
            # horizontal
            data[n_pxls + i, :, :, i] = 1
            # diagonal
            data[-2, :, i, i] = 1
            data[-1, :, n_pxls - 1 - i, i] = 1
        self.data = data.float()
        self.targets = self.data.long()
        # self.data -= self.data.mean(dim=(0, 2, 3))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class genMNIST(datasets.MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.targets = self.data[:, None, :, :].long()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class MNISTexperiment(PytorchExperiment):
    def setup(self):
        self.elog.print("Config:")
        self.elog.print(self.config)

        # # Prepare datasets (colored MNIST)
        # self.dataset_train = ColoredMNIST('mnist-hw1.pkl', train=True, transform=transforms.ToTensor())
        # self.dataset_test = ColoredMNIST('mnist-hw1.pkl', train=False, transform=transforms.ToTensor())

        # Prepare datasets (normal MNIST)
        self.dataset_train = genMNIST('.', train=True, download=True, transform=transforms.ToTensor(),
                                      # transform=transforms.Compose([
                                      #     transforms.ToTensor(), transforms.Normalize((0.1307,), (1.,))])
                                      )
        self.dataset_test = genMNIST('.', train=False, download=True, transform=transforms.ToTensor(),
                                     # transform=transforms.Compose([
                                     #     transforms.ToTensor(), transforms.Normalize((0.1307,), (1.,))])
                                     )

        data_loader_kwargs = {'num_workers': 1, 'pin_memory': True} if self.config.use_cuda else {}
        self.train_data_loader = torch.utils.data.DataLoader(
            self.dataset_train, batch_size=self.config.batch_size, shuffle=True, **data_loader_kwargs)
        self.test_data_loader = torch.utils.data.DataLoader(
            self.dataset_test, batch_size=self.config.batch_size, shuffle=True, **data_loader_kwargs)

        self.device = torch.device("cuda" if self.config.use_cuda else "cpu")

        # # colored MNIST
        # self.model = PixelCNN(h=120)
        # normal MNIST
        self.model = PixelCNN(h=64, img_channels=1, discrete_channels=256)
        # self.model = SimplePixelCNN(7 * [64], img_channels=1, discrete_channels=256)

        self.model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.loss_fn = nn.NLLLoss(reduction='none')

        self.save_checkpoint(name="checkpoint_start")
        self.elog.print('Experiment set up.')

    def train(self, epoch):
        self.model.train()
        # loop over batches
        for batch_idx, b in enumerate(self.train_data_loader):
            x, targets = b
            # x = torch.zeros_like(x)
            if self.config.use_cuda:
                x = x.cuda()
                targets = targets.cuda()
            pixelwise_logp = self.model(x)
            pixelwise_loss = self.loss_fn(pixelwise_logp, targets)
            loss = pixelwise_loss.sum() / x.shape[0]

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch_idx % self.config.log_interval == 0:
                reported_loss = loss.item() / np.log(2) / np.prod(x.shape[1:])
                # plot train loss
                self.add_result(value=reported_loss, name='Train_Loss',
                                counter=epoch + batch_idx / len(self.train_data_loader), tag='Loss')
                # log train batch loss and progress
                self.clog.show_text(
                    'Train Epoch: {} [{}/{} samples ({:.0f}%)]\t Batch Loss: {:.6f}'.format(
                        epoch, batch_idx * len(x), len(self.train_data_loader.dataset),
                               100. * batch_idx / len(self.train_data_loader), reported_loss), name="log")

                self.clog.show_image_grid(
                    x, name="mnist_training", n_iter=epoch + batch_idx / len(self.train_data_loader),
                    iter_format="{:0.02f}")
                self.save_checkpoint(name="checkpoint", n_iter=batch_idx)

    def validate(self, epoch):
        self.model.eval()

        with torch.no_grad():
            validation_loss = 0
            for batch_idx, b in enumerate(self.test_data_loader):
                x, targets = b
                if self.config.use_cuda:
                    x = x.cuda()
                    targets = targets.cuda()
                if validation_loss == 0:
                    n_dim = np.prod(x.shape[1:])

                pixelwise_logp = self.model(x)
                pixelwise_loss = self.loss_fn(pixelwise_logp, targets)
                validation_loss += pixelwise_loss.sum()
            validation_loss /= len(self.test_data_loader.dataset)

            # get some samples
            samples = self.model.sample(5, (28, 28), device=self.device)

            reported_loss = validation_loss.item() / np.log(2) / n_dim
        # plot the test loss
        self.add_result(value=reported_loss, name='Validation_Loss', counter=epoch + 1, tag='Loss')

        # log validation loss and accuracy
        self.elog.print('\nValidation set: Average loss: {:.4f})\n'.format(reported_loss))

        if samples.ndim == 4:
            samples = np.transpose(samples, (0, 2, 3, 1))
        self.clog.show_image_grid(
            samples, name="Samples", n_iter=epoch + batch_idx / len(self.train_data_loader),
            iter_format="{:0.02f}")
        self.save_checkpoint(name="checkpoint", n_iter=batch_idx)


class BaselineExperiment(MNISTexperiment):
    def setup(self):
        self.elog.print("Config:")
        self.elog.print(self.config)

        # Prepare dataset
        self.dataset_train = ToyDataset()
        self.dataset_test = ToyDataset()

        # self.dataset_train = torch.utils.data.Subset(self.dataset_train, range(4))
        # self.dataset_test = torch.utils.data.Subset(self.dataset_test, range(4, 8))

        data_loader_kwargs = {'num_workers': 1, 'pin_memory': True} if self.config.use_cuda else {}
        self.train_data_loader = torch.utils.data.DataLoader(
            self.dataset_train, batch_size=self.config.batch_size, shuffle=True, **data_loader_kwargs)
        self.test_data_loader = torch.utils.data.DataLoader(
            self.dataset_test, batch_size=self.config.batch_size, shuffle=True, **data_loader_kwargs)

        self.device = torch.device("cuda" if self.config.use_cuda else "cpu")

        n_hidden = 100
        self.model = SimplePixelCNN(5 * [n_hidden], img_channels=1, discrete_channels=2)

        self.model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.loss_fn = nn.NLLLoss(reduction='none')

        self.save_checkpoint(name="checkpoint_start")
        self.elog.print('Experiment set up.')


class coloredMNISTexperiment(MNISTexperiment):
    def setup(self):
        self.elog.print("Config:")
        self.elog.print(self.config)

        # Prepare datasets (colored MNIST)
        self.dataset_train = ColoredMNIST('mnist-hw1.pkl', train=True, transform=transforms.ToTensor())
        self.dataset_test = ColoredMNIST('mnist-hw1.pkl', train=False, transform=transforms.ToTensor())

        data_loader_kwargs = {'num_workers': 1, 'pin_memory': True} if self.config.use_cuda else {}
        self.train_data_loader = torch.utils.data.DataLoader(
            self.dataset_train, batch_size=self.config.batch_size, shuffle=True, **data_loader_kwargs)
        self.test_data_loader = torch.utils.data.DataLoader(
            self.dataset_test, batch_size=self.config.batch_size, shuffle=True, **data_loader_kwargs)

        self.device = torch.device("cuda" if self.config.use_cuda else "cpu")

        # colored MNIST
        self.model = PixelCNN(h=120)

        self.model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.loss_fn = nn.NLLLoss(reduction='none')

        self.save_checkpoint(name="checkpoint_start")
        self.elog.print('Experiment set up.')
