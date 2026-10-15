#!/usr/bin/env python3
"""Generate Index,Hogwarts House predictions using a saved model."""
import argparse
import csv
import sys
from pathlib import Path
from data import records
from model import load_model, predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset')
    parser.add_argument('weights')
    parser.add_argument('--output', default='houses.csv')
    args = parser.parse_args()
    try:
        rows = records(args.dataset)
        if 'Index' not in rows[0]:
            raise ValueError('Missing Index column.')
        indices = [row['Index'] for row in rows]
        if any(not index for index in indices) or len(set(indices)) != len(indices):
            raise ValueError('Index values must be nonempty and unique.')
        predictions = predict(rows, load_model(args.weights))
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['Index', 'Hogwarts House'])
            writer.writerows(zip(indices, predictions))
        print(f'Saved {len(predictions)} predictions: {args.output}')
    except (OSError, ValueError, csv.Error) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
