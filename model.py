import os
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


def _project_to_l2_ball(z):
    return z / np.maximum(np.sqrt(np.sum(z**2, axis=1))[:, np.newaxis], 1)


def _generate_latent_from_pca(train_loader, z_dim):
    print("[Latent Init] Preparing PCA")
    indices, images = zip(*[
        (indices, images) for indices, images in train_loader])
    indices, images = torch.cat(indices), torch.cat(images)

    print("[Latent Init] Performing the actual PCA")
    pca = PCA(n_components=z_dim)
    pca.fit(images.view(images.size()[0], -1).numpy())

    print("[Latent Init] Creating and populating the latent variables")
    Z = np.empty((len(train_loader.dataset), z_dim))
    Z = torch.tensor(
        pca.transform(images.view(images.size()[0], -1).numpy()),
        requires_grad=True).float()
    return Z


def _get_latent_variables(train_loader, z_dim):
    _path = '/tmp/GLO_pca_init_{}_{}.pt'.format(
        train_loader.dataset.base.filename, z_dim)
    if os.path.isfile(_path):
        print(
            '[Latent Init] PCA already calculated before and saved at {}'.
            format(_path))
        Z = torch.load(_path)
    else:
        Z = _generate_latent_from_pca(train_loader, z_dim)
        torch.save(Z, _path)
    return Z


class LatentVariables(nn.Module):
    def __init__(self, train_loader, z_dim=100):
        super(LatentVariables, self).__init__()
        self.Z = Parameter(_get_latent_variables(train_loader, z_dim))

    def forward(self, indices):
        return self.Z[indices]


class Generator(nn.Module):
    """Vanilla DCGAN generator

    Copied from https://github.com/pytorch/examples/blob/master/dcgan/main.py
    Minor adaptation to match the 32x32 dimension on CIFAR10"""

    def __init__(self, train_loader, z_dim=100, n_filters=64):
        super(Generator, self).__init__()

        self.z_dim = z_dim
        index, image = train_loader.dataset[0]
        out_channels, out_width, out_height = image.size()

        assert out_width in [32, 64]
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(z_dim, n_filters * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(n_filters * 8),
            nn.ReLU(True),
            # state size. (n_filters*8) x 4 x 4
            nn.ConvTranspose2d(
                n_filters * 8, n_filters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters * 4),
            nn.ReLU(True),
            # state size. (n_filters*4) x 8 x 8
            nn.ConvTranspose2d(
                n_filters * 4, n_filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters * 2),
            nn.ReLU(True),
            # state size. (n_filters*2) x 16 x 16
            # nn.ConvTranspose2d(
            #     n_filters * 2, n_filters, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(n_filters),
            # nn.ReLU(True),
            # state size. (n_filters) x 32 x 32
            # nn.ConvTranspose2d(n_filters, out_channels, 4, 2, 1, bias=False),
            nn.ConvTranspose2d(
                n_filters * 2, out_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (out_channels) x 64 x 64
        )

    def forward(self, code):
        return self.main(code.view(code.size(0), self.z_dim, 1, 1))


class CombinedModel(nn.Module):
    def __init__(self, train_loader, z_dim=100, n_filters=64):
        super(CombinedModel, self).__init__()
        self.Z = LatentVariables(train_loader, z_dim)
        self.Generator = Generator(train_loader, z_dim)

    def forward(self, index):
        code = self.Z(index)
        # code = code.view(code.size(0), self.z_dim, 1, 1)
        return self.Generator(code)
