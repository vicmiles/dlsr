#!/usr/bin/env python3
"""Compare a predictions file with true houses using sklearn's accuracy_score."""
import argparse
import csv
import sys
from sklearn.metrics import accuracy_score
from common.data import records


def houses_by_index(path):
    rows = records(path)
    for name in ('Index', 'Hogwarts House'):
        if name not in rows[0]:
            raise ValueError(f'{path}: missing {name} column.')
    return {row['Index']: row['Hogwarts House'] for row in rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions', help='For example houses.csv.')
    parser.add_argument('truth', help='CSV with Index and the real Hogwarts House.')
    args = parser.parse_args()
    try:
        predicted, truth = houses_by_index(args.predictions), houses_by_index(args.truth)
        if set(predicted) != set(truth):
            raise ValueError('Index values in both files must be the same.')
        indices = list(truth)
        if any(not truth[i] for i in indices):
            raise ValueError(f'{args.truth}: some Hogwarts House values are empty.')
        y_true = [truth[i] for i in indices]
        y_pred = [predicted[i] for i in indices]
        score = accuracy_score(y_true, y_pred)
        print(f'Accuracy: {score:.4f} ({sum(t == p for t, p in zip(y_true, y_pred))}/{len(indices)})')
        print('Passed: at least 0.98' if score >= 0.98 else 'Failed: below 0.98')
    except (OSError, ValueError, csv.Error) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
