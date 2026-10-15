#!/usr/bin/env python3
"""Holdout evaluation: fit preprocessing only on the training partition."""
import argparse
import csv
import json
import sys
from pathlib import Path
from data import records
from model import stratified_split, train, evaluate
from logreg_train import add_training_arguments, training_options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset')
    parser.add_argument('--validation-size', type=float, default=0.2)
    parser.add_argument('--output', help='Optional JSON report.')
    add_training_arguments(parser)
    args = parser.parse_args()
    try:
        training, validation = stratified_split(records(args.dataset, True), args.validation_size, args.seed)
        model = train(training, **training_options(args))
        report = evaluate(validation, model)
        report.update({'seed': args.seed, 'training_size': len(training), 'training': model['training']})
        print(f"Holdout accuracy: {report['accuracy']:.2%} ({report['correct']}/{report['total']})")
        print('Confusion matrix: rows = actual, columns = predicted')
        print('Order: ' + ', '.join(report['classes']))
        for house, row in zip(report['classes'], report['confusion_matrix']):
            print(f'{house:12s} {row}')
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    except (OSError, ValueError, csv.Error) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
