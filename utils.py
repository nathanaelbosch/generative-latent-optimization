import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset


class IndexToImageDataset(Dataset):
    """Wrap a dataset to map indices to images

    In other words, instead of producing (X, y) it produces (idx, X). The label
    y is not relevant for our task.
    """
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]
        return (idx, img)


def gaussian(x, sigma=1.0):
    return np.exp(-(x**2) / (2*(sigma**2)))

def build_gauss_kernel(
        size=5, sigma=1.0, out_channels=1, in_channels=1, device=None):
    """Construct the convolution kernel for a gaussian blur

    See https://en.wikipedia.org/wiki/Gaussian_blur for a definition.
    Overall I first generate a NxNx2 matrix of indices, and then use those to
    calculate the gaussian function on each element. The two dimensional
    Gaussian function is then the product along axis=2.
    """
    if size % 2 != 1:
        raise ValueError("kernel size must be uneven")
    grid = np.mgrid[range(size), range(size)] - size//2
    kernel = np.prod(gaussian(grid, sigma), axis=0)
    kernel /= np.sum(kernel)

    # repeat same kernel for all pictures and all channels
    # Also, conv weight should be (out_channels, in_channels/groups, h, w)
    kernel = np.tile(kernel, (out_channels, in_channels, 1, 1))
    kernel = torch.from_numpy(kernel).to(torch.float).to(device)
    return kernel


def blur_images(images, kernel):
    """Convolve the gaussian kernel with the given stack of images"""
    n_channels, _, kw, kh = kernel.shape
    img = F.pad(images, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
    return F.conv2d(img, kernel)


def laplacian_pyramid(images, kernel, max_levels=5):
    """Laplacian pyramid of each image

    https://en.wikipedia.org/wiki/Pyramid_(image_processing)#Laplacian_pyramid
    """
    current = images
    pyramid = []

    for level in range(max_levels):
        filtered = blur_images(current, kernel)
        diff = current - filtered
        pyramid.append(diff)
        current = F.avg_pool2d(filtered, 2)
    pyramid.append(current)
    return pyramid


class LapLoss(nn.Module):
    def __init__(self, max_levels=5, kernel_size=5, sigma=1.0):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.kernel_size = kernel_size
        self.sigma = sigma
        self._gauss_kernel = None

    def forward(self, output, target):
        if (self._gauss_kernel is None
                or self._gauss_kernel.shape[1] != output.shape[1]):
            self._gauss_kernel = build_gauss_kernel(
                out_channels=output.shape[1],
                in_channels=output.shape[1],
                device=output.device)
        output_pyramid = laplacian_pyramid(
            output, self._gauss_kernel, max_levels=self.max_levels)
        target_pyramid = laplacian_pyramid(
            target, self._gauss_kernel, max_levels=self.max_levels)
        diff_levels = [F.l1_loss(o, t)
                        for o, t in zip(output_pyramid, target_pyramid)]
        loss = sum([2**(-2*j) * diff_levels[j]
                    for j in range(self.max_levels)])
        return loss
