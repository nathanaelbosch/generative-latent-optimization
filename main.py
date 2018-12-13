from collections import defaultdict
import os
import logging

import torch
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter
import tqdm
import numpy as np
from ignite.engine import Events, create_supervised_trainer
from ignite.metrics import RunningAverage, Loss

from model import Generator
from utils import IndexToImageDataset, pca_init, LapLoss


def setup_logger(level='DEBUG'):
    """Setup personal logger and its handler for this module

    All of this is necessary in order to only get the debug messages from this
    module, otherwise I get tons of messages from all possible third party
    imports.
    """
    logger = logging.getLogger(__name__)
    hdlr = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt='%(levelname)s|%(name)s|%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(level)
    return logger


logger = setup_logger()
writer = SummaryWriter(comment='')


def get_dataloader(dataset, batch_size, data_path='./data'):
    if dataset.lower() == 'mnist':
        data_transforms = [
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ]

        train_data = datasets.MNIST(
            data_path,
            train=True,
            download=True,
            transform=transforms.Compose(data_transforms))
    elif dataset.lower() == 'cifar10':
        data_transforms = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]

        train_data = datasets.CIFAR10(
            data_path,
            train=True,
            download=True,
            transform=transforms.Compose(data_transforms))

    train_loader = torch.utils.data.DataLoader(
        IndexToImageDataset(train_data),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2)
    return train_loader


def log_images(images, epoch, tag='image'):
    writer.add_image(
        tag, torchvision.utils.make_grid(images, nrow=5, normalize=True),
        epoch)


def main(
        # 'use_cuda': True and torch.cuda.is_available(),
        data_path: ('Path where the dataset is stored', 'option') = './data/',
        seed=1,
        model_learning_rate: ('', 'option') = 1,
        model_momentum: ('', 'option') = 0,
        latent_learning_rate: ('', 'option') = 10,
        latent_momentum: ('', 'option') = 0,
        batch_size: ('', 'option') = 128,
        # test_batch_size: ('', 'option')=1000,
        latent_size: ('', 'option') = 100,
        epochs: ('', 'option') = 1000,
        dataset: ('', 'option') = 'cifar10',
        tensorboard_description: ('', 'option') = '',
        no_cuda: ('Do not use CUDA, even if available.', 'flag',
                  'no-cuda') = False,
):
    use_cuda = torch.cuda.is_available() and not no_cuda
    device = torch.device('cuda' if use_cuda else 'cpu')
    if tensorboard_description != '':
        tensorboard_description = '_' + tensorboard_description
    torch.manual_seed(seed)

    train_loader = get_dataloader(dataset, batch_size, data_path)

    # Get dimensions
    index, image = train_loader.dataset[0]
    channels, width, height = image.size()
    logger.debug('image shape: {}'.format(image.size()))

    _path = '/tmp/GLO_{}_pca_latent_init.pt'.format(dataset)
    if os.path.isfile(_path):
        logger.info('PCA already calculated before - Using local file {}'.
                    format(_path))
        Z = torch.load(_path)
    else:
        Z = pca_init(train_loader, latent_size, device=device)
        torch.save(Z, '/tmp/GLO_{}_pca_latent_init.pt'.format(dataset))

    model = Generator(Z, out_channels=channels).to(device)

    # Log graph to tensorboard
    dummy_index = torch.ones([1, 1], dtype=torch.int64, device=device)
    writer.add_graph(model, dummy_index)

    # Get test indices
    test_indices = torch.randint(
        len(train_loader.dataset), size=(10, )).to(torch.int64).to(device)
    test_images = [train_loader.dataset[int(i)][1] for i in test_indices]
    test_images = torch.cat(
        [x.view(1, *x.size()) for x in test_images]).to(device)
    log_images(test_images, 0, 'original_image')

    optimizer = optim.Adam([
        {
            'params': model.main.parameters(),
            # 'lr': model_learning_rate,
            # 'momentum': model_momentum,
        },
        {
            'params': (model.Z, ),
            # 'lr': latent_learning_rate,
            # 'momentum': latent_momentum,
        },
    ])

    loss_fn = LapLoss(max_levels=3)
    trainer = create_supervised_trainer(
        model, optimizer, loss_fn, device=device)

    desc = '[Epoch {:d}/{:d}] Loss: {:.4f}'
    log_interval = 1
    pbar = tqdm.tqdm(total=len(train_loader), desc=desc.format(0, epochs, 0))

    @trainer.on(Events.EPOCH_STARTED)
    def initialize_running_loss(engine):
        engine.state.running_loss = 0
        engine.state._running_loss_sum = 0

    @trainer.on(Events.ITERATION_COMPLETED)
    def calculate_running_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1
        engine.state._running_loss_sum += engine.state.output
        engine.state.running_loss = engine.state._running_loss_sum / iter

    @trainer.on(Events.ITERATION_COMPLETED)
    def update_progress_bar(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1

        if iter % log_interval == 0:
            pbar.desc = desc.format(engine.state.epoch, epochs,
                                    engine.state.running_loss)
            pbar.update(log_interval)

    @trainer.on(Events.EPOCH_COMPLETED)
    def refresh_progress_bar(engine):
        pbar.n = pbar.last_print_n = 0

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_loss(engine):
        epoch = engine.state.epoch
        writer.add_scalar('metrics/train_loss', engine.state.running_loss,
                          epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_model_parameters(engine):
        epoch = engine.state.epoch
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            writer.add_histogram(tag, value.data.cpu().numpy(), epoch)
            writer.add_histogram(tag + '/grad',
                                 value.grad.data.cpu().numpy(), epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_reconstructed_images(engine):
        output = model(test_indices)
        log_images(output, engine.state.epoch, 'reconstructed_image')

    trainer.run(train_loader, max_epochs=epochs)


if __name__ == '__main__':
    import plac
    plac.call(main)
    # main()
