#!/usr/bin/env python3
"""Train and save four binary classifiers with shared preprocessing."""
import argparse
import csv
import sys
from data import records
from model import train, save_model


def add_training_arguments(parser):
    parser.add_argument('--optimizer', choices=['batch', 'sgd', 'mini-batch'], default='batch')
    parser.add_argument('--epochs', type=int, default=2500)
    parser.add_argument('--learning-rate', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--regularization', type=float, default=0.0001)
    parser.add_argument('--seed', type=int, default=42)


def training_options(args):
    return {key: getattr(args, key) for key in (
        'optimizer', 'epochs', 'learning_rate', 'batch_size', 'regularization', 'seed')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset')
    parser.add_argument('--output', default='weights.json')
    add_training_arguments(parser)
    args = parser.parse_args()
    try:
        rows = records(args.dataset, require_labels=True)
        model = train(rows, **training_options(args))
        save_model(model, args.output)
        print(f'Trained on {len(rows)} students using {args.optimizer}.')
        print(f"Loss: {model['history'][0]['loss']:.6f} -> {model['history'][-1]['loss']:.6f}")
        print(f'Saved model: {args.output}')
    except (OSError, ValueError, csv.Error) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
