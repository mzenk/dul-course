import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from trixi.experiment import PytorchExperiment
from models import RealNVP2D, RealNVP
from layers import SimpleResnet
import pickle
from datasets import CelebLQ, Dequantize


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


class SmileyExperiment(PytorchExperiment):
    def setup(self):
        self.elog.print("Config:")
        self.elog.print(self.config)

        # Prepare datasets
        data, targets = sample_data()
        n_train = int(.8 * len(data))
        self.dataset_train = torch.tensor(data[:n_train], dtype=torch.float32)
        self.dataset_test = torch.tensor(data[n_train:], dtype=torch.float32)

        data_loader_kwargs = {'num_workers': 1, 'pin_memory': True} if self.config.use_cuda else {}
        self.train_data_loader = DataLoader(
            self.dataset_train, batch_size=self.config.batch_size, shuffle=True, **data_loader_kwargs)
        self.test_data_loader = DataLoader(
            self.dataset_test, batch_size=self.config.batch_size, shuffle=True, **data_loader_kwargs)

        self.device = torch.device("cuda" if self.config.use_cuda else "cpu")

        self.model = RealNVP2D(n_coupling=self.config.n_coupling, prior=self.config.prior)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        self.save_checkpoint(name="checkpoint_start")
        self.elog.print('Experiment set up.')

    def train(self, epoch):
        self.model.train()
        # loop over batches
        for batch_idx, x in enumerate(self.train_data_loader):
            self.optimizer.zero_grad()
            if self.config.use_cuda:
                x = x.cuda()
            _, ll = self.model(x)
            loss = - ll.mean()

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

                # self.clog.show_image_grid(
                #     x, name="training minibatch", n_iter=epoch + batch_idx / len(self.train_data_loader),
                #     iter_format="{:0.02f}")
                self.save_checkpoint(name="checkpoint", n_iter=batch_idx)

    def validate(self, epoch):
        self.model.eval()

        with torch.no_grad():
            validation_loss = 0
            for batch_idx, x in enumerate(self.test_data_loader):
                self.optimizer.zero_grad()
                n_dim = x.shape[1]
                if self.config.use_cuda:
                    x = x.cuda()
                _, ll = self.model(x)
                validation_loss += - ll.sum()
            validation_loss /= len(self.test_data_loader.dataset)

            # # get some samples
            # samples = self.model.sample(5, (28, 28), device=self.device)

        # if samples.ndim == 4:
        #     samples = np.transpose(samples, (0, 2, 3, 1))
        # self.clog.show_image_grid(
        #     samples, name="Samples", n_iter=epoch + batch_idx / len(self.train_data_loader),
        #     iter_format="{:0.02f}")

        reported_loss = validation_loss.item() / np.log(2) / n_dim
        # plot the test loss
        self.add_result(value=reported_loss, name='Validation_Loss', counter=epoch + 1, tag='Loss')

        # log validation loss and accuracy
        self.elog.print('\nValidation set: Average loss: {:.4f})\n'.format(reported_loss))

        self.save_checkpoint(name="checkpoint", n_iter=batch_idx)


class CelebAExperiment(PytorchExperiment):
    def setup(self):
        self.elog.print("Config:")
        self.elog.print(self.config)

        # Prepare datasets
        self.dataset_train = CelebLQ('hw2_q2.pkl', train=True, transform=Dequantize(3))
        self.dataset_test = CelebLQ('hw2_q2.pkl', train=False, transform=Dequantize(3))

        data_loader_kwargs = {'num_workers': 1, 'pin_memory': True} if self.config.use_cuda else {}
        self.train_data_loader = DataLoader(
            self.dataset_train, batch_size=self.config.batch_size, shuffle=True, **data_loader_kwargs)
        self.test_data_loader = DataLoader(
            self.dataset_test, batch_size=self.config.batch_size, shuffle=True, **data_loader_kwargs)

        self.device = torch.device("cuda" if self.config.use_cuda else "cpu")

        self.model = RealNVP((3, 32, 32), n_coupling=self.config.n_coupling, n_filters=self.config.n_filters)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        self.save_checkpoint(name="checkpoint_start")
        self.elog.print('Experiment set up.')

    def train(self, epoch):
        self.model.train()
        # loop over batches
        for batch_idx, x in enumerate(self.train_data_loader):
            self.optimizer.zero_grad()
            if self.config.use_cuda:
                x = x.cuda()
            _, ll = self.model(x)
            loss = - ll.mean()

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
                    x, name="training minibatch", n_iter=epoch + batch_idx / len(self.train_data_loader),
                    iter_format="{:0.02f}")
                self.save_checkpoint(name="checkpoint", n_iter=batch_idx)

    def validate(self, epoch):
        self.model.eval()

        with torch.no_grad():
            validation_loss = 0
            n_data = 0
            for batch_idx, x in enumerate(self.test_data_loader):
                self.optimizer.zero_grad()
                if self.config.use_cuda:
                    x = x.cuda()
                _, ll = self.model(x)
                validation_loss += - ll.sum()
                n_data += len(x)
            validation_loss /= n_data

            # # get some samples
            # samples = self.model.sample(5, (28, 28), device=self.device)

        # if samples.ndim == 4:
        #     samples = np.transpose(samples, (0, 2, 3, 1))
        # self.clog.show_image_grid(
        #     samples, name="Samples", n_iter=epoch + batch_idx / len(self.train_data_loader),
        #     iter_format="{:0.02f}")

        reported_loss = validation_loss.item() / np.log(2) / np.prod(self.model.img_dim)
        # plot the test loss
        self.add_result(value=reported_loss, name='Validation_Loss', counter=epoch + 1, tag='Loss')

        # log validation loss and accuracy
        self.elog.print('\nValidation set: Average loss: {:.4f})\n'.format(reported_loss))

        self.save_checkpoint(name="checkpoint", n_iter=batch_idx)


class MNISTExperiment(PytorchExperiment):
    def setup(self):
        self.elog.print("Config:")
        self.elog.print(self.config)

        # Prepare datasets
        transform = transforms.Compose(
            [transforms.ToTensor(),
             Dequantize(255)]
        )
        self.dataset_train = datasets.MNIST(root="data/", download=True, transform=transform, train=True)
        self.dataset_test = datasets.MNIST(root="data/", download=True, transform=transform, train=False)

        try:
            self.dataset_train = torch.utils.data.Subset(self.dataset_train, np.arange(self.config.subset_size))
            self.dataset_test = torch.utils.data.Subset(self.dataset_test, np.arange(self.config.subset_size))
        except AttributeError:
            pass

        data_loader_kwargs = {'num_workers': 1, 'pin_memory': True} if self.config.use_cuda else {}
        self.train_data_loader = DataLoader(
            self.dataset_train, batch_size=self.config.batch_size, shuffle=True, **data_loader_kwargs)
        self.test_data_loader = DataLoader(
            self.dataset_test, batch_size=self.config.batch_size, shuffle=True, **data_loader_kwargs)

        self.device = torch.device("cuda" if self.config.use_cuda else "cpu")

        self.model = RealNVP((1, 28, 28), n_coupling=self.config.n_coupling, n_filters=self.config.n_filters)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

        self.save_checkpoint(name="checkpoint_start")
        self.elog.print('Experiment set up.')

    def train(self, epoch):
        self.model.train()
        # loop over batches
        for batch_idx, (x, _) in enumerate(self.train_data_loader):
            self.optimizer.zero_grad()
            if self.config.use_cuda:
                x = x.cuda()
            _, ll = self.model(x)
            loss = - ll.mean()

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
                    x, name="training minibatch", n_iter=epoch + batch_idx / len(self.train_data_loader),
                    iter_format="{:0.02f}")
                self.save_checkpoint(name="checkpoint", n_iter=batch_idx)

    def validate(self, epoch):
        self.model.eval()

        with torch.no_grad():
            validation_loss = 0
            n_data = 0
            for batch_idx, (x, _) in enumerate(self.test_data_loader):
                self.optimizer.zero_grad()
                if self.config.use_cuda:
                    x = x.cuda()
                _, ll = self.model(x)
                validation_loss += - ll.sum()
                n_data += len(x)
            validation_loss /= n_data

            # # get some samples
            # samples = self.model.sample(5, (28, 28), device=self.device)

        # if samples.ndim == 4:
        #     samples = np.transpose(samples, (0, 2, 3, 1))
        # self.clog.show_image_grid(
        #     samples, name="Samples", n_iter=epoch + batch_idx / len(self.train_data_loader),
        #     iter_format="{:0.02f}")

        reported_loss = validation_loss.item() / np.log(2) / np.prod(self.model.img_dim)
        # plot the test loss
        self.add_result(value=reported_loss, name='Validation_Loss', counter=epoch + 1, tag='Loss')

        # log validation loss and accuracy
        self.elog.print('\nValidation set: Average loss: {:.4f})\n'.format(reported_loss))

        self.save_checkpoint(name="checkpoint", n_iter=batch_idx)


class MNIST_classification(PytorchExperiment):
    def setup(self):

        self.elog.print("Config:")
        self.elog.print(self.config)

        ### Get Dataset
        transf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.dataset_train = datasets.MNIST(root="experiment_dir/data/", download=True,
                                            transform=transf, train=True)
        self.dataset_test = datasets.MNIST(root="experiment_dir/data/", download=True,
                                           transform=transf, train=False)

        data_loader_kwargs = {'num_workers': 1, 'pin_memory': True} if self.config.use_cuda else {}

        self.train_data_loader = torch.utils.data.DataLoader(self.dataset_train, batch_size=self.config.batch_size,
                                                             shuffle=True, **data_loader_kwargs)
        self.test_data_loader = torch.utils.data.DataLoader(self.dataset_test, batch_size=self.config.batch_size,
                                                            shuffle=True, **data_loader_kwargs)

        self.device = torch.device("cuda" if self.config.use_cuda else "cpu")

        self.model = SimpleResnet(1, n_filters=10, n_blocks=4)
        self.model.to(self.device)

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.learning_rate,
                                   momentum=self.config.momentum)

        self.save_checkpoint(name="checkpoint_start")
        self.vlog.plot_model_structure(self.model,
                                       [self.config.batch_size, 1, 28, 28],
                                       name='Model Structure')

        self.batch_counter = 0
        self.elog.print('Experiment set up.')

    def train(self, epoch):

        self.model.train()

        for batch_idx, (data, target) in enumerate(self.train_data_loader):

            self.batch_counter += 1

            if self.config.use_cuda:
                data, target = data.cuda(), target.cuda()

            self.optimizer.zero_grad()

            output = self.model(data)
            self.loss = F.cross_entropy(output, target)
            self.loss.backward()

            self.optimizer.step()

            if batch_idx % self.config.log_interval == 0:
                # plot train loss (mathematically mot 100% correct, just so that lisa can sleep at night (if no one is breathing next to her ;-P) )
                self.add_result(value=self.loss.item(), name='Train_Loss',
                                counter=epoch + batch_idx / len(self.train_data_loader), tag='Loss')
                # log train batch loss and progress
                self.clog.show_text(
                    'Train Epoch: {} [{}/{} samples ({:.0f}%)]\t Batch Loss: {:.6f}'
                        .format(epoch, batch_idx * len(data),
                                len(self.train_data_loader.dataset),
                                100. * batch_idx / len(self.train_data_loader),
                                self.loss.item()), name="log")

                self.clog.show_image_grid(data, name="mnist_training",
                                          n_iter=epoch + batch_idx / len(self.train_data_loader),
                                          iter_format="{:0.02f}")

                self.save_checkpoint(name="checkpoint", n_iter=batch_idx)

    def validate(self, epoch):
        self.model.eval()

        validation_loss = 0
        correct = 0

        for data, target in self.test_data_loader:
            if self.config.use_cuda:
                data, target = data.cuda(), target.cuda()
            output = self.model(data)
            validation_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
        validation_loss /= len(self.test_data_loader.dataset)
        # plot the test loss
        self.add_result(value=validation_loss, name='Validation_Loss',
                        counter=epoch + 1, tag='Loss')
        # plot the test accuracy
        acc = 100. * correct / len(self.test_data_loader.dataset)
        self.add_result(value=acc, name='ValidationAccurracy',
                        counter=epoch + 1, tag='Accurracy')

        # log validation loss and accuracy
        self.elog.print(
            '\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
                .format(validation_loss, correct, len(self.test_data_loader.dataset),
                        100. * correct / len(self.test_data_loader.dataset)))
