#!/usr/bin/env python3

import argparse
import csv
import sys
from pathlib import Path

# Direct script execution adds this folder, not the project root, to sys.path.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from common.data import read_dataset, numeric_columns
from common.statistics import basic_statistics, bonus_statistics


def print_basic_statistics(columns, bonus=False):
    names = list(columns)
    if not names:
        raise ValueError('No numerical columns found.')
    # Print groups to keep a wide dataset readable in a terminal.
    for start in range(0, len(names), 4):
        group = names[start:start + 4]
        results = {}
        for name in group:
            results[name] = basic_statistics(columns[name])
            if bonus:
                results[name].update(bonus_statistics(columns[name]))
        widths = []
        for name in group:
            cells = ['N/A' if v is None else f'{v:.6f}' for v in results[name].values()]
            widths.append(max(15, len(name), *(len(cell) for cell in cells)) + 2)
        print('Statistic'.ljust(12) + ''.join(n.rjust(w) for n, w in zip(group, widths)))
        for key in results[group[0]]:
            cells = []
            for name, width in zip(group, widths):
                value = results[name][key]
                cells.append(('N/A' if value is None else f'{value:.6f}').rjust(width))
            print(key.ljust(12) + ''.join(cells))
        print()


def main():
    parser = argparse.ArgumentParser(description='Descriptive statistics calculated by hand.')
    parser.add_argument('dataset')
    parser.add_argument('--bonus', action='store_true', help='Add variance, range and IQR.')
    args = parser.parse_args()
    try:
        header, rows = read_dataset(args.dataset)
        print_basic_statistics(numeric_columns(header, rows), args.bonus)
    except (OSError, ValueError, csv.Error) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
