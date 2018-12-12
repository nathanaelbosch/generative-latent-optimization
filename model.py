import torch.nn as nn
import torch.nn.functional as F

# needs z_dim, n_filters, out_channels
# z_dim = size of latent vector default 100
# n_filters default 64, but in the paper 128
# out_channels = 3 => out_channels?

class Generator(nn.Module):
    """DCGAN generator

    Copied from https://github.com/pytorch/examples/blob/master/dcgan/main.py
    Minor adaptation to match the 32x32 dimension on CIFAR10"""
    def __init__(self, z_dim=100, n_filters=64, out_channels=3, out_width=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        assert out_width in [32, 64]
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(z_dim, n_filters * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(n_filters * 8),
            nn.ReLU(True),
            # state size. (n_filters*8) x 4 x 4
            nn.ConvTranspose2d(n_filters * 8, n_filters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters * 4),
            nn.ReLU(True),
            # state size. (n_filters*4) x 8 x 8
            nn.ConvTranspose2d(n_filters * 4, n_filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters * 2),
            nn.ReLU(True),
            # state size. (n_filters*2) x 16 x 16
            # nn.ConvTranspose2d(n_filters * 2, n_filters, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(n_filters),
            # nn.ReLU(True),
            # state size. (n_filters) x 32 x 32
            # nn.ConvTranspose2d(n_filters, out_channels, 4, 2, 1, bias=False),
            nn.ConvTranspose2d(n_filters * 2, out_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (out_channels) x 64 x 64
        )

    def forward(self, code):
        return self.main(code.view(code.size(0), self.z_dim, 1, 1))
