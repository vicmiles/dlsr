#!/usr/bin/env python3
"""Pairwise course scatter plots with score histograms on the diagonal."""
import csv
import sys
import numpy as np
from data import COURSES, FEATURES, HOUSES
from describe import basic_statistics
from plotting import arguments, setup, load, values, pairs, finish, COLORS


def main():
    parser = arguments(__doc__, 'plots/pair_plot.png')
    parser.add_argument('--selected', action='store_true', help='Only show the selected model features.')
    args = parser.parse_args()
    try:
        rows = load(args.dataset)
        plt = setup(args.no_show)
        courses = FEATURES if args.selected else COURSES
        n = len(courses)
        figure, axes = plt.subplots(n, n, figsize=(2 * n, 2 * n))
        for i, first in enumerate(courses):
            for j, second in enumerate(courses):
                ax = axes[i, j]
                if i == j:
                    groups = {house: values(rows, first, house) for house in HOUSES}
                    stats = basic_statistics([v for group in groups.values() for v in group])
                    if not stats['Count']:
                        raise ValueError(f'No numeric observations for {first}.')
                    low, high = stats['Min'], stats['Max']
                    if low == high:
                        low, high = low - 0.5, high + 0.5
                    bins = np.linspace(low, high, 20)
                    for house, group in groups.items():
                        if group:
                            ax.hist(group, bins=bins, density=True, histtype='step', color=COLORS[house])
                else:
                    for house in HOUSES:
                        x, y = pairs(rows, second, first, house)
                        ax.scatter(x, y, s=3, alpha=0.3, color=COLORS[house], rasterized=True, edgecolors='none')
                ax.tick_params(labelsize=6)
                if i == n - 1:
                    ax.set_xlabel(second.replace(' ', '\n'), fontsize=8)
                else:
                    ax.tick_params(labelbottom=False)
                if j == 0:
                    ax.set_ylabel(first.replace(' ', '\n'), fontsize=8)
                else:
                    ax.tick_params(labelleft=False)
        from matplotlib.lines import Line2D
        figure.legend([Line2D([], [], marker='o', linestyle='', color=COLORS[h]) for h in HOUSES],
                      HOUSES, loc='upper center', ncol=4, fontsize=14, bbox_to_anchor=(0.5, 0.985))
        figure.suptitle('Which courses separate the houses?', fontsize=22, y=0.998)
        figure.subplots_adjust(wspace=0.12, hspace=0.12, top=0.96, bottom=0.09, left=0.08)
        finish(plt, figure, args)
        print('Selected model features: ' + ', '.join(FEATURES))
    except (OSError, ValueError, csv.Error) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
