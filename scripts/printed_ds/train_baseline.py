#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Training launcher."""
import sys
import torch
import configargparse

import numpy as np

from tqdm import tqdm
from pathlib import Path
from pprint import pprint
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from printed_dataset import PrintedDataset

sys.path.append(str(Path(sys.path[0]).parent))
from common.utils import build_alphabet, create_model, calc_cer


def get_args():
    """Arguments parser."""
    parser = configargparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', is_config_file=True,
                        help='Path to config file.')

    # dataset
    parser.add_argument('--train-dir', type=Path, required=True,
                        help='Path to train dir.')
    parser.add_argument('--valid-dir', type=Path, required=True,
                        help='Path to validation dir.')

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


def save_model(model, optim, args, ep, metrics, best_metrics, models_dir):
    """Save best models and update best metrics."""
    save_data = {'model': model.state_dict(),
                 'optim': optim.state_dict(),
                 'args': args,
                 'epoch': ep,
                 'valid_metrics': metrics,
                 'i2c': model.i2c}

    for m in metrics.keys():
        for stage in ['train', 'valid']:
            if metrics[m][stage] < best_metrics[m][stage]:
                model_path = models_dir.joinpath(stage + '-' + m + '.pth')
                torch.save(save_data, model_path)
                best_metrics[m][stage] = metrics[m][stage]

                print(f'Saved {stage} {m}')


def epoch_step(model, loaders, device, optim):
    metrics = {'cer': {'train': 0.0, 'valid': 0.0},
               'cer-beam': {'train': 0.0, 'valid': 0.0},
               'loss': {'train': 0.0, 'valid': 0.0}}

    for stage in ['train', 'valid']:
        is_train = stage == 'train'
        model.train() if is_train else model.eval()
        loader_size = len(loaders[stage])

        torch.set_grad_enabled(is_train)
        for i, (img, text, lens) in enumerate(tqdm(loaders[stage], desc=stage)):
            if is_train:
                optim.zero_grad()

            # forward
            img, text, lens = img.to(device), text.to(device), lens.to(device)
            logits, log_probs = model(img)

            # loss
            loss = model.calc_loss(logits, log_probs, text, lens)
            metrics['loss'][stage] += loss.item() / loader_size

            # cer
            gt_text = loaders[stage].dataset.tensor2text(text)
            pd_beam, pd_lens = model.decode(log_probs)
            pd_beam = loaders[stage].dataset.tensor2text(pd_beam)

            gt_lens = lens.detach().cpu().numpy()
            cer = calc_cer(gt_text, pd_beam, gt_lens)
            metrics['cer-beam'][stage] += cer / loader_size

            preds = torch.argmax(logits, dim=1).permute(1, 0)
            pd_text = loaders[stage].dataset.tensor2text(preds.permute(1, 0))
            cer = calc_cer(gt_text, pd_text, gt_lens)
            metrics['cer'][stage] += cer / loader_size

            # print
            if i == 0:
                gt_text = gt_text[0][:lens[0]]
                pd_text = loaders[stage].dataset.tensor2text(preds[:, 0][:lens[0]])
                pd_beam = pd_beam[0][:lens[0]]

                print('\nGT:', gt_text)
                print('PD:', pd_text)
                print('PD beam:', pd_beam)

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

    # model
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    c2i, i2c = build_alphabet()
    model = create_model(c2i, i2c).to(device)
    optim = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    # datasets
    text_max_len = 62
    ds_args = {'c2i': c2i,
               'i2c': i2c,
               'text_max_len': text_max_len}
    ds_train = PrintedDataset(args.train_dir, **ds_args)
    ds_valid = PrintedDataset(args.valid_dir, **ds_args)

    dl_args = {'batch_size': args.bs,
               'num_workers': args.workers,
               'shuffle': True}
    loaders = {'train': DataLoader(ds_train, **dl_args),
               'valid': DataLoader(ds_valid, **dl_args)}

    # model saving initialization
    best_metrics = {'cer': {'train': np.inf, 'valid': np.inf},
                    'cer-beam': {'train': np.inf, 'valid': np.inf},
                    'loss': {'train': np.inf, 'valid': np.inf}}

    ep = 1
    while ep != args.epochs + 1:
        print(f'\nEpoch #{ep}')
        metrics = epoch_step(model, loaders, device, optim)

        # dump metrics
        for m_name, m_values in metrics.items():
            writer.add_scalars(m_name, m_values, ep)

        # dump best metrics
        for m_name, m_values in best_metrics.items():
            writer.add_scalars('best_' + m_name, m_values, ep)

        # print metrics
        print('Current metrics:')
        pprint(metrics)

        print('Best metrics:')
        pprint(best_metrics)

        # save best models and update best metrics
        save_model(model, optim, args, ep, metrics, best_metrics, models_dir)
        writer.flush()

        ep += 1


if __name__ == '__main__':
    main()