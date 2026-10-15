"""Regression tests for statistics, mathematical gradients and the CLI pipeline."""
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

import numpy as np

from data import HOUSES, records, read_dataset, numeric_columns
from describe import basic_statistics, bonus_statistics, print_basic_statistics
from model import (evaluate, fit_preprocessing, load_model, loss_and_gradient,
                   predict, save_model, sigmoid, stratified_split, train, transform)

ROOT = Path(__file__).resolve().parents[1]


class StatisticsTests(unittest.TestCase):
    def test_known_values(self):
        stats = basic_statistics([1, 2, 3, 4])
        self.assertEqual(stats['Count'], 4)
        self.assertEqual(stats['Mean'], 2.5)
        self.assertAlmostEqual(stats['Std'], (5 / 3) ** 0.5)
        self.assertEqual([stats[k] for k in ('Min', '25%', '50%', '75%', 'Max')], [1, 1.75, 2.5, 3.25, 4])
        bonus = bonus_statistics([1, 2, 3, 4])
        self.assertAlmostEqual(bonus['Variance'], 5 / 3)
        self.assertEqual(bonus['Range'], 3)
        self.assertEqual(bonus['IQR'], 1.5)

    def test_missing_singleton_and_text(self):
        cols = numeric_columns(['number', 'empty', 'text'], [['5', '', 'a'], ['', '', 'b']])
        self.assertEqual(cols, {'number': [5.0], 'empty': []})
        output = io.StringIO()
        with redirect_stdout(output):
            print_basic_statistics(cols, bonus=True)
        self.assertIn('N/A', output.getvalue())
        self.assertIsNone(basic_statistics([])['Mean'])

    def test_csv_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.csv'
            for text in ('', 'a,a\n1,2\n', 'a,b\n1\n', 'a,b\n'):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    read_dataset(path)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = records(ROOT / 'datasets/dataset_train.csv', True)

    def test_gradient_against_finite_differences(self):
        rng = np.random.default_rng(9)
        x = np.column_stack((np.ones(6), rng.normal(size=(6, 3))))
        y = (rng.random((6, 4)) > 0.5).astype(float)
        weights = rng.normal(size=(4, 4))
        _, analytical = loss_and_gradient(x, y, weights, 0.03)
        numerical = np.zeros_like(weights)
        epsilon = 1e-6
        for index in np.ndindex(weights.shape):
            above, below = weights.copy(), weights.copy()
            above[index] += epsilon
            below[index] -= epsilon
            numerical[index] = (loss_and_gradient(x, y, above, 0.03)[0] -
                                loss_and_gradient(x, y, below, 0.03)[0]) / (2 * epsilon)
        np.testing.assert_allclose(analytical, numerical, atol=1e-8, rtol=1e-6)

    def test_sigmoid_extreme_values(self):
        with np.errstate(over='raise', invalid='raise'):
            np.testing.assert_allclose(sigmoid(np.array([-10000., 0, 10000.])), [0, 0.5, 1])

    def test_preprocessing_uses_training_only(self):
        preprocessing = fit_preprocessing([{'x': '1'}, {'x': '3'}, {'x': ''}], ['x'])
        self.assertEqual(preprocessing['centers'], [2])
        result = transform([{'x': ''}, {'x': '100'}], preprocessing)
        self.assertEqual(result[0, 1], 0)
        self.assertAlmostEqual(result[1, 1], 98 / (2 ** 0.5))
        self.assertEqual(preprocessing['centers'], [2])
        with self.assertRaises(ValueError):
            fit_preprocessing([{'x': ''}], ['x'])

    def test_all_optimizers_on_holdout(self):
        training, validation = stratified_split(self.rows)
        self.assertFalse(set(row['Index'] for row in training) & set(row['Index'] for row in validation))
        for optimizer, epochs in [('batch', 2500), ('mini-batch', 250), ('sgd', 100)]:
            with self.subTest(optimizer=optimizer):
                model = train(training, optimizer=optimizer, epochs=epochs)
                self.assertGreaterEqual(evaluate(validation, model)['accuracy'], 0.98)
                self.assertLess(model['history'][-1]['loss'], model['history'][0]['loss'])

    def test_save_reload_and_invalid_model(self):
        model = train(self.rows, epochs=20)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'weights.json'
            save_model(model, path)
            self.assertEqual(predict(self.rows[:30], model), predict(self.rows[:30], load_model(path)))
            model['preprocessing']['scales'][0] = 0
            path.write_text(json.dumps(model))
            with self.assertRaises(ValueError):
                load_model(path)

    def test_cli_pipeline_and_output_format(self):
        with tempfile.TemporaryDirectory() as directory:
            weights, output = Path(directory) / 'weights.json', Path(directory) / 'houses.csv'
            def run(*args):
                return subprocess.run([sys.executable, *map(str, args)], cwd=ROOT,
                                      capture_output=True, text=True)
            trained = run('logreg_train.py', 'datasets/dataset_train.csv', '--output', weights)
            self.assertEqual(trained.returncode, 0, trained.stderr)
            predicted = run('logreg_predict.py', 'datasets/dataset_test.csv', weights, '--output', output)
            self.assertEqual(predicted.returncode, 0, predicted.stderr)
            with output.open() as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(reader.fieldnames, ['Index', 'Hogwarts House'])
                rows = list(reader)
            source = records(ROOT / 'datasets/dataset_test.csv')
            self.assertEqual([r['Index'] for r in rows], [r['Index'] for r in source])
            self.assertEqual(len(rows), 400)
            self.assertTrue(all(r['Hogwarts House'] in HOUSES for r in rows))
            failed = run('logreg_predict.py', 'datasets/dataset_test.csv', Path(directory) / 'missing.json')
            self.assertNotEqual(failed.returncode, 0)
            self.assertNotIn('Traceback', failed.stderr)


if __name__ == '__main__':
    unittest.main()
