#!/usr/bin/env python3

import csv
import math
import sys


def read_dataset(path):
    """Return the CSV header and all data rows."""
    with open(path, "r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader)
        rows = []
        for row in reader:
            rows.append(row)
    return header, rows


def to_number(value):
    """Convert a non-empty value to float; return None if it is not numeric."""
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def numeric_columns(header, rows):
    """Collect usable numbers from columns whose values are numeric."""
    columns = {}

    for column_index, column_name in enumerate(header):
        values = []
        has_number = False
        has_text = False

        for row in rows:
            value = to_number(row[column_index])
            if value is None:
                if row[column_index] != "":
                    has_text = True
            else:
                has_number = True
                values.append(value)

        if has_number and not has_text:
            columns[column_name] = values

    return columns


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


def print_basic_statistics(columns):
    """Print the currently implemented statistics for every numeric column."""
    for name, values in columns.items():
        stats = basic_statistics(values)
        print(name)
        print(f"  Count: {stats['Count']}")
        print(f"  Mean:  {stats['Mean']:.6f}")
        print(f"  Std:   {stats['Std']:.6f}")
        print(f"  Min:   {stats['Min']:.6f}")
        print(f"  25%:   {stats['25%']:.6f}")
        print(f"  50%:   {stats['50%']:.6f}")
        print(f"  75%:   {stats['75%']:.6f}")
        print(f"  Max:   {stats['Max']:.6f}")


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} dataset.csv", file=sys.stderr)
        return 1

    try:
        header, rows = read_dataset(sys.argv[1])
    except (OSError, StopIteration) as error:
        print(f"Error: cannot read dataset: {error}", file=sys.stderr)
        return 1

    columns = numeric_columns(header, rows)
    print_basic_statistics(columns)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
