#!/usr/bin/env python3

import argparse
import csv
import math
import sys

from data import read_dataset, to_number, numeric_columns


def percentile(sorted_values, fraction):
    """Return a percentile using linear interpolation between neighbours."""
    count = 0
    for _ in sorted_values:
        count += 1
    if count == 0:
        return None
    if count == 1:
        return sorted_values[0]

    position = (count - 1) * fraction
    lower_index = int(position)
    upper_index = lower_index + 1

    if upper_index >= count:
        return sorted_values[lower_index]

    distance = position - lower_index
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]

    return lower_value + (upper_value - lower_value) * distance


def basic_statistics(values):
    """Calculate basic descriptive statistics without helper functions."""
    count = 0
    total = 0.0
    minimum = None
    maximum = None

    for value in values:
        count += 1
        total += value

        if minimum is None or value < minimum:
            minimum = value
        if maximum is None or value > maximum:
            maximum = value

    if count == 0:
        return {
            "Count": 0,
            "Mean": None,
            "Std": None,
            "Min": None,
            "25%": None,
            "50%": None,
            "75%": None,
            "Max": None,
        }

    mean = total / count
    squared_difference_sum = 0.0

    for value in values:
        difference = value - mean
        squared_difference_sum += difference * difference

    if count > 1:
        variance = squared_difference_sum / (count - 1)
        standard_deviation = math.sqrt(variance)
    else:
        standard_deviation = None

    sorted_values = sorted(values)

    return {
        "Count": count,
        "Mean": mean,
        "Std": standard_deviation,
        "Min": minimum,
        "25%": percentile(sorted_values, 0.25),
        "50%": percentile(sorted_values, 0.50),
        "75%": percentile(sorted_values, 0.75),
        "Max": maximum,
    }


def bonus_statistics(values):
    stats = basic_statistics(values)
    variance = stats['Std'] ** 2 if stats['Std'] is not None else None
    return {
        'Variance': variance,
        'Range': stats['Max'] - stats['Min'] if values else None,
        'IQR': stats['75%'] - stats['25%'] if values else None,
    }


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
