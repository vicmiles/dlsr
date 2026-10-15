#!/usr/bin/env python3
"""Visualize two nearly redundant features (their relation is negative)."""
import csv
import sys
from data import COURSES, HOUSES
from plotting import arguments, setup, load, pairs, finish, COLORS


def main():
    parser = arguments(__doc__, 'plots/scatter_plot.png')
    parser.add_argument('--x', choices=COURSES, default='Astronomy')
    parser.add_argument('--y', choices=COURSES, default='Defense Against the Dark Arts')
    args = parser.parse_args()
    try:
        rows = load(args.dataset)
        plt = setup(args.no_show)
        figure, ax = plt.subplots(figsize=(9, 6))
        for house in HOUSES:
            x, y = pairs(rows, args.x, args.y, house)
            ax.scatter(x, y, s=15, alpha=0.55, color=COLORS[house], label=house, edgecolors='none')
        ax.set(xlabel=args.x, ylabel=args.y, title='Similar information can have opposite signs')
        ax.legend(title='Hogwarts House')
        ax.grid(alpha=0.15)
        figure.tight_layout()
        finish(plt, figure, args)
    except (OSError, ValueError, csv.Error) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
