"""Shared plotting helpers. Plot libraries draw only; statistics are ours."""
import argparse
import os
from pathlib import Path

from data import COURSES, HOUSES, records, to_number

COLORS = {'Gryffindor': '#b73535', 'Hufflepuff': '#d29a19',
          'Ravenclaw': '#3776ab', 'Slytherin': '#34845e'}


def arguments(description, default_output):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('dataset', nargs='?', default='datasets/dataset_train.csv')
    parser.add_argument('--output', default=default_output)
    parser.add_argument('--no-show', action='store_true', help='Save without opening a window.')
    return parser


def setup(no_show):
    # Keep Matplotlib caches inside the project and support headless execution.
    os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent / '.mplconfig'))
    os.environ.setdefault('XDG_CACHE_HOME', str(Path(__file__).resolve().parent / '.mplconfig'))
    import matplotlib
    if no_show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False,
                         'font.size': 10, 'figure.facecolor': 'white'})
    return plt


def load(path):
    rows = records(path, require_labels=True)
    for course in COURSES:
        if course not in rows[0]:
            raise ValueError(f'Missing course: {course}')
    return rows


def values(rows, course, house):
    return [number for row in rows if row['Hogwarts House'] == house
            if (number := to_number(row[course])) is not None]


def pairs(rows, first, second, house):
    points = []
    for row in rows:
        if row['Hogwarts House'] != house:
            continue
        a, b = to_number(row[first]), to_number(row[second])
        if a is not None and b is not None:
            points.append((a, b))
    return [p[0] for p in points], [p[1] for p in points]


def finish(plt, figure, args):
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f'Saved plot: {args.output}')
    if not args.no_show:
        plt.show()
    plt.close(figure)
