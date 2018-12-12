from collections import defaultdict
import logging

import torch
from torch.autograd import Variable
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter
import tqdm
import numpy as np

from model import Generator
from utils import IndexedDataset, pca_init, LapLoss

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

writer = SummaryWriter(comment='')


def get_dataloader(dataset, batch_size, data_path='./data'):
    if dataset.lower() == 'mnist':
        data_transforms = [transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))]

        train_data = datasets.MNIST(
            data_path, train=True, download=True,
            transform=transforms.Compose(data_transforms))
    elif dataset.lower() == 'cifar10':
        data_transforms = [transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        train_data = datasets.CIFAR10(
            data_path, train=True, download=True,
            transform=transforms.Compose(data_transforms))

    train_loader = torch.utils.data.DataLoader(
        IndexedDataset(train_data),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2)
    return train_loader


def train_epoch(model, device, train_loader, optimizer, Z, epoch, epochs):
    """Train the model for a single epoch on the train data"""
    running = defaultdict(int)

    model.train()
    train_bar = tqdm.tqdm(train_loader)
    dataset_size = len(train_loader.dataset)
    loss_fn = LapLoss(max_levels=3)
    for data, _, indices in train_bar:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(Z[indices])
        # logger.debug('Output shape: {}'.format(output.size()))
        loss = loss_fn(output, data)
        loss.backward()
        optimizer.step()

        batch_size = data.size(0)
        running['total_seen'] += batch_size
        running['loss_total'] += loss * batch_size
        running['loss_avg'] = running['loss_total'] / running['total_seen']

        train_bar.set_description(
            desc=('[Epoch {:d}/{:d}] Loss: {:.4f}').format(
                epoch, epochs,
                running['loss_avg'],
            ))

    writer.add_scalar('metrics/train_loss', running['loss_avg'], epoch)
    # writer.add_scalar('metrics/train_acc', running['acc'], epoch)
    writer.add_image(
        'reconstructed_image',
        torchvision.utils.make_grid(output[:10], nrow=2, normalize=True),
        epoch
    )
    writer.add_image(
        'original_image',
        torchvision.utils.make_grid(data[:10], nrow=2, normalize=True),
        epoch
    )


def project_l2_ball(z):
    return z / np.maximum(np.sqrt(np.sum(z**2, axis=1))[:, np.newaxis], 1)


def main(
        # 'use_cuda': True and torch.cuda.is_available(),
        data_path: ('Path where the dataset is stored',
                    'option')='./data/',
        seed=1,
        model_learning_rate: ('', 'option') = 1,
        model_momentum: ('', 'option')=0,
        latent_learning_rate: ('', 'option')=10,
        latent_momentum: ('', 'option')=0,
        batch_size: ('', 'option')=128,
        # test_batch_size: ('', 'option')=1000,
        latent_size: ('', 'option')=100,
        epochs: ('', 'option')=1000,
        dataset: ('', 'option')='cifar10',
):
    use_cuda = torch.cuda.is_available()

    torch.manual_seed(seed)
    device = torch.device('cuda' if use_cuda else 'cpu')
    train_loader  = get_dataloader(dataset, batch_size, data_path)

    # Get dimensions
    image, _, _ = train_loader.dataset[0]
    channels, width, height = image.size()
    logger.debug('image shape: {}'.format(image.size()))

    model = Generator(out_channels=channels).to(device)

    Z = pca_init(train_loader, latent_size)
    Z = project_l2_ball(Z)
    Z = Variable(torch.from_numpy(Z).float().to(device), requires_grad=True)

    writer.add_graph(model, (Z[:10], ))

    optimizer = optim.Adam([
        {'params': model.parameters(),
         # 'lr': model_learning_rate,
         # 'momentum': model_momentum,
        },
        {'params': (Z, ),
         # 'lr': latent_learning_rate,
         # 'momentum': latent_momentum,
        },
    ])

    for epoch in range(1, epochs + 1):
        train_epoch(
            model, device, train_loader, optimizer,
            Z, epoch, epochs)


if __name__ == '__main__':
    import plac
    plac.call(main)
    # main()
