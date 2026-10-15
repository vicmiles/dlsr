"""CSV parsing shared by analysis, visualization and classification."""
import csv
import math

HOUSES = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
COURSES = [
    'Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts',
    'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic',
    'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms', 'Flying',
]
# Exclude weak separators and one member of the near-duplicate Astronomy/DADA pair.
FEATURES = [name for name in COURSES if name not in (
    'Arithmancy', 'Care of Magical Creatures', 'Defense Against the Dark Arts')]


def read_dataset(path):
    with open(path, newline='', encoding='utf-8-sig') as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if not header or any(not name.strip() for name in header):
            raise ValueError('CSV must have a nonempty header.')
        if len(set(header)) != len(header):
            raise ValueError('Duplicate CSV column names.')
        rows = []
        for line, row in enumerate(reader, 2):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) != len(header):
                raise ValueError(f'CSV line {line}: expected {len(header)} fields, got {len(row)}.')
            rows.append([value.strip() for value in row])
    if not rows:
        raise ValueError('Dataset has no data rows.')
    return header, rows


def to_number(value):
    if not value.strip():
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def numeric_columns(header, rows):
    columns = {}
    for index, name in enumerate(header):
        values, has_text = [], False
        for row in rows:
            raw = row[index]
            number = to_number(raw)
            if number is not None:
                values.append(number)
            elif raw and raw.lower() not in ('nan', 'inf', '-inf', '+inf', 'infinity', '-infinity', '+infinity'):
                has_text = True
        if not has_text:
            columns[name] = values
    return columns


def records(path, require_labels=False):
    header, rows = read_dataset(path)
    result = [dict(zip(header, row)) for row in rows]
    if require_labels:
        if 'Hogwarts House' not in header:
            raise ValueError('Missing Hogwarts House column.')
        for line, row in enumerate(result, 2):
            if row['Hogwarts House'] not in HOUSES:
                raise ValueError(f'CSV line {line}: missing or unknown Hogwarts House.')
    return result
