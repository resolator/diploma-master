#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Print metadata from the checkpoint file."""
import torch
import argparse
from pathlib import Path
from pprint import pprint


def get_args():
    """Arguments parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt-path', type=Path, required=True,
                        help='Path to the checkpoint.pth file.')

    return parser.parse_args()


def main():
    """Application entry point."""
    args = get_args()

    ckpt = torch.load(args.ckpt_path, map_location='cpu')

    print('=' * 79)
    print('Epoch number:', ckpt['epoch'])
    print('Metrics for this model:')
    pprint(ckpt['metrics'])

    print('\nBest metrics for this training:')
    pprint(ckpt['best_metrics'])

    print('\nArguments:')
    pprint(vars(ckpt['args']))


if __name__ == '__main__':
    main()
