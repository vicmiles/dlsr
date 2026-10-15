#!/usr/bin/env python3
"""Compare normalized score distributions across all four houses."""
import csv
import sys
import numpy as np
from data import COURSES, HOUSES
from describe import basic_statistics
from plotting import arguments, setup, load, values, finish, COLORS


def main():
    parser = arguments(__doc__, 'plots/histogram.png')
    parser.add_argument('--course', choices=COURSES, help='Show only one course.')
    args = parser.parse_args()
    try:
        rows = load(args.dataset)
        plt = setup(args.no_show)
        courses = [args.course] if args.course else COURSES
        columns = 1 if args.course else 3
        height = (len(courses) + columns - 1) // columns
        figure, axes = plt.subplots(height, columns, figsize=(7 if args.course else 16, height * 3.1), squeeze=False)
        for ax, course in zip(axes.flat, courses):
            grouped = {house: values(rows, course, house) for house in HOUSES}
            stats = basic_statistics([value for group in grouped.values() for value in group])
            if stats['Count'] == 0:
                raise ValueError(f'No numeric observations for {course}.')
            low, high = stats['Min'], stats['Max']
            if low == high:
                low, high = low - 0.5, high + 0.5
            bins = np.linspace(low, high, 25)
            for house, group in grouped.items():
                if group:
                    ax.hist(group, bins=bins, density=True, histtype='step', linewidth=1.8,
                            color=COLORS[house], label=house)
            ax.set_title(course, fontsize=11)
            ax.set_xlabel('Score')
            ax.set_ylabel('Density')
        for ax in list(axes.flat)[len(courses):]:
            ax.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.965))
        figure.suptitle('Course distributions by Hogwarts house', fontsize=17, y=0.995)
        figure.tight_layout(rect=(0, 0, 1, 0.94))
        finish(plt, figure, args)
        print('For the supplied dataset: Care of Magical Creatures has strongly overlapping distributions.')
    except (OSError, ValueError, csv.Error) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
