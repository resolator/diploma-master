#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Training launcher."""
import torch
import configargparse

import numpy as np

from tqdm import tqdm
from pathlib import Path
from pprint import pprint
from iam_dataset import IAMDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def get_args():
    """Arguments parser."""
    parser = configargparse.ArgumentParser(description=__doc__)

    # dataset
    parser.add_argument('--images-dir', type=Path, required=True,
                        help='Path to root dir with images (ex. iam/lines/).')
    parser.add_argument('--mkp-file', type=Path, required=True,
                        help='Path to markup file (ex. iam/ascii/lines.txt).')
    parser.add_argument('--train-split', type=Path, required=True,
                        help='Path to train split file. Can be generated '
                             'using scripts/utils/gen_split_file.py.')
    parser.add_argument('--valid-split', type=Path, required=True,
                        help='Path to validation split file. Can be generated '
                             'using scripts/utils/gen_split_file.py.')

    parser.add_argument('--epochs', type=int, default=-1,
                        help='Number of epochs to train. '
                             'Pass -1 for infinite training.')
    parser.add_argument('--bs', type=int, default=16,
                        help='Batch size.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate.')
    parser.add_argument('--workers', type=int, default=2,
                        help='Number of data loader workers.')

    parser.add_argument('--save-to', type=Path,
                        help='Path to save dir.')

    return parser.parse_args()


def create_model():
    return 0


def save_model(model, optim, args, ep, metrics, best_metrics, models_dir):
    """Save best models and update best metrics."""
    save_data = {'model': model.state_dict(),
                 'optim': optim.state_dict(),
                 'args': args,
                 'epoch': ep,
                 'valid_metrics': metrics}

    for stage in ['train', 'valid']:
        # save by loss
        if metrics['loss'][stage] <= best_metrics['loss'][stage]:
            model_path = models_dir.joinpath(stage + 'loss.pth')
            torch.save(save_data, model_path)
            best_metrics['loss'][stage] = metrics['loss'][stage]

            print(f'Saved {stage} loss')

        # save by acc
        if metrics['acc'][stage] >= best_metrics['acc'][stage]:
            model_path = models_dir.joinpath(stage + 'acc.pth')
            torch.save(save_data, model_path)
            best_metrics['acc'][stage] = metrics['acc'][stage]

            print(f'Saved {stage} acc')


def epoch_step(model, loaders, device, loss_fn, optim):
    metrics = {'acc': {'train': 0.0, 'valid': 0.0},
               'loss': {'train': 0.0, 'valid': 0.0}}

    for stage in ['train', 'valid']:
        is_train = stage == 'train'
        model.train() if is_train else model.eval()
        loader_size = len(loaders[stage])

        torch.set_grad_enabled(is_train)
        for imgs, lbls in tqdm(loaders[stage], desc=stage):
            if is_train:
                optim.zero_grad()

            # forward
            imgs, lbls = imgs.to(device), lbls.to(device)
            preds = model(imgs)

            # loss
            loss = loss_fn(preds, lbls)
            metrics['loss'][stage] += loss.item() / loader_size

            # accuracy
            lbls_pred = preds.max(dim=1)[1]
            acc = np.sum(lbls_pred == lbls, dtype=float) / len(lbls)
            metrics['acc'][stage] += acc.item() / loader_size

            # backward
            if is_train:
                loss.backward()
                optim.step()

    return metrics


def main():
    """Application entry point."""
    args = get_args()

    # make all dirs
    models_dir = args.save_to.joinpath('models')
    models_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = args.save_to.joinpath('logs')
    logs_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(logs_dir)

    # datasets
    ds_train = IAMDataset(args.images_dir, args.mkp_file, args.train_split)
    ds_valid = IAMDataset(args.images_dir, args.mkp_file, args.valid_split)

    loaders = {
        'train': DataLoader(ds_train, args.bs, True, num_workers=args.workers),
        'valid': DataLoader(ds_valid, args.bs, num_workers=args.workers)
    }

    # model
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    model = create_model().do(device)
    optim = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    # model saving initialization
    best_metrics = {'acc': {'train': 0.0, 'valid': 0.0},
                    'loss': {'train': np.inf, 'valid': np.inf}}

    ep = 1
    while ep != args.epochs + 1:
        print(f'\nEpoch #{ep}')
        metrics = epoch_step(model, loaders, device, loss_fn, optim)

        # dump metrics
        for m_name, m_values in metrics.items():
            writer.add_scalars(m_name, m_values, ep)

        # print metrics
        print('Current metrics:')
        pprint(metrics)

        print('Best metrics:')
        pprint(best_metrics)

        # save best models and update best metrics
        save_model(model, optim, args, ep, metrics, best_metrics, models_dir)

        ep += 1


if __name__ == '__main__':
    main()
