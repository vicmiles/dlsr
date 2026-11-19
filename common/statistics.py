"""Hand-written statistics shared by analysis and preprocessing."""
import math


def merge_sort(values):
    """Return a new ascending list using merge sort."""
    values = list(values)
    if len(values) <= 1:
        return values

    middle = len(values) // 2
    left = merge_sort(values[:middle])
    right = merge_sort(values[middle:])

    merged = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged


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

    sorted_values = merge_sort(values)

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


