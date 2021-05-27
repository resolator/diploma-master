#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Generate train/validation/test split file."""
import argparse

from pathlib import Path
from random import shuffle


def get_args():
    """Arguments parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images-dir', type=Path, required=True,
                        help='Path to root dir with images (ex. iam/lines/).')
    parser.add_argument('--save-to', type=Path, required=True,
                        help='Path to save dir.')
    parser.add_argument('--valid-ratio', type=float, default=0.15,
                        help='Validation part size.')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                        help='Test part size.')

    return parser.parse_args()


def save_file(names, stage, save_to):
    """Simple array saver to file."""
    file_path = save_to.joinpath(stage + '.txt')
    with open(file_path, 'w') as f:
        [print(x, file=f) for x in names]

    print(stage.title() + ' saved to:', file_path.absolute())


def main():
    """Application entry point."""
    args = get_args()

    # read and shuffle images names
    names = [x.stem for x in args.images_dir.rglob('*.png')]
    shuffle(names)

    # split them
    n = len(names)
    valid_n = int(n * args.valid_ratio)
    test_n = int(n * args.test_ratio)
    train_n = n - valid_n - test_n

    # select names
    valid_names = names[:valid_n] if valid_n > 0 else []
    test_names = names[valid_n:valid_n + test_n] if test_n > 0 else []
    train_names = names[-train_n:] if train_n > 0 else []

    # save
    args.save_to.mkdir(exist_ok=True, parents=True)
    print()
    save_file(valid_names, 'valid', args.save_to)
    save_file(test_names, 'test', args.save_to)
    save_file(train_names, 'train', args.save_to)


if __name__ == '__main__':
    main()
