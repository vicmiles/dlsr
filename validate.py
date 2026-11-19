#!/usr/bin/env python3
"""Check a saved model on a dataset that already has Hogwarts House labels."""
import argparse
import csv
import json
import sys
from pathlib import Path
from common.data import records
from model import load_model, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset')
    parser.add_argument('weights')
    parser.add_argument('--output', help='Optional JSON report.')
    args = parser.parse_args()
    try:
        report = evaluate(records(args.dataset, True), load_model(args.weights))
        print(f"Accuracy: {report['accuracy']:.2%} ({report['correct']}/{report['total']})")
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
