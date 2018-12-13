import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
from sklearn.decomposition import PCA


class IndexToImageDataset(Dataset):
    """Wrapper for a dataset in order to also return indices

    In other words, instead of producing (X, y) it produces (X, y, idx)
    """
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        return (idx, img)


def pca_init(train_loader, z_dim, device=None):
    n_pca = 64*64*3*2

    # first, take a subset of train set to fit the PCA
    X_pca = np.vstack([
        X.cpu().numpy().reshape(len(X), -1)
        for i, (_, X)
        in zip(tqdm(
            range(n_pca // train_loader.batch_size),
            'collect data for PCA'),
               train_loader)
    ])
    print("perform PCA...")
    pca = PCA(n_components=z_dim)
    pca.fit(X_pca)
    # then, initialize latent vectors to the pca projections of the complete dataset
    Z = np.empty((len(train_loader.dataset), z_dim))
    for idx, X in tqdm(train_loader, 'pca projection'):
        Z[idx] = pca.transform(X.cpu().numpy().reshape(len(X), -1))
    Z = project_l2_ball(Z)
    Z = Variable(torch.from_numpy(Z).float().to(device), requires_grad=True)
    return Z


def build_gauss_kernel(size=5, sigma=1.0, n_channels=1, cuda=False):
    if size % 2 != 1:
        raise ValueError("kernel size must be uneven")
    grid = np.float32(np.mgrid[0:size,0:size].T)
    gaussian = lambda x: np.exp((x - size//2)**2/(-2*sigma**2))**2
    kernel = np.sum(gaussian(grid), axis=2)
    kernel /= np.sum(kernel)
    # repeat same kernel across depth dimension
    kernel = np.tile(kernel, (n_channels, 1, 1))
    # conv weight should be (out_channels, groups/in_channels, h, w),
    # and since we have depth-separable convolution we want the groups dimension to be 1
    kernel = torch.FloatTensor(kernel[:, None, :, :])
    if cuda:
        kernel = kernel.cuda()
    return Variable(kernel, requires_grad=False)


def conv_gauss(img, kernel):
    """ convolve img with a gaussian kernel that has been built with build_gauss_kernel """
    n_channels, _, kw, kh = kernel.shape
    img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
    return F.conv2d(img, kernel, groups=n_channels)


def laplacian_pyramid(img, kernel, max_levels=5):
    current = img
    pyr = []

    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        diff = current - filtered
        pyr.append(diff)
        current = F.avg_pool2d(filtered, 2)

    pyr.append(current)
    return pyr


class LapLoss(nn.Module):
    def __init__(self, max_levels=5, k_size=5, sigma=2.0):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.k_size = k_size
        self.sigma = sigma
        self._gauss_kernel = None

    def forward(self, input, target):
        if self._gauss_kernel is None or self._gauss_kernel.shape[1] != input.shape[1]:
            self._gauss_kernel = build_gauss_kernel(
                size=self.k_size, sigma=self.sigma,
                n_channels=input.shape[1], cuda=input.is_cuda
            )
        pyr_input  = laplacian_pyramid( input, self._gauss_kernel, self.max_levels)
        pyr_target = laplacian_pyramid(target, self._gauss_kernel, self.max_levels)
        return sum(F.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))


def project_l2_ball(z):
    return z / np.maximum(np.sqrt(np.sum(z**2, axis=1))[:, np.newaxis], 1)
