"""One-vs-rest logistic regression; NumPy is used only for array arithmetic."""
import json
import math
from pathlib import Path

import numpy as np

from data import FEATURES, HOUSES, to_number
from describe import basic_statistics


def matrix(rows, features):
    for name in features:
        if name not in rows[0]:
            raise ValueError(f'Missing feature: {name}')
    result = []
    for line, row in enumerate(rows, 2):
        values = []
        for name in features:
            raw = row[name]
            value = to_number(raw)
            if value is None and raw and raw.lower() not in ('nan',):
                raise ValueError(f'CSV line {line}: invalid number for {name}: {raw!r}')
            values.append(float('nan') if value is None else value)
        result.append(values)
    return np.asarray(result, dtype=float)


def fit_preprocessing(rows, features):
    raw = matrix(rows, features)
    centers, scales = [], []
    for j, name in enumerate(features):
        observed = [float(value) for value in raw[:, j] if math.isfinite(value)]
        if not observed:
            raise ValueError(f'Feature {name} has no observed training values.')
        stats = basic_statistics(observed)
        centers.append(stats['Mean'])
        scales.append(stats['Std'] if stats['Std'] and stats['Std'] > 1e-12 else 1.0)
    return {'features': list(features), 'centers': centers, 'scales': scales}


def transform(rows, preprocessing):
    raw = matrix(rows, preprocessing['features'])
    centers = np.asarray(preprocessing['centers'])
    scales = np.asarray(preprocessing['scales'])
    # Missing observations become the training mean (zero after standardization).
    raw = np.where(np.isnan(raw), centers, raw)
    scaled = (raw - centers) / scales
    return np.column_stack((np.ones(len(rows)), scaled))


def sigmoid(z):
    # This equivalent expression avoids exp(large positive number) overflow.
    return np.exp(-np.logaddexp(0.0, -z))


def loss_and_gradient(x, y, weights, regularization=0.0):
    scores = x @ weights
    loss = np.sum(np.logaddexp(0.0, scores) - y * scores) / len(x)
    gradient = x.T @ (sigmoid(scores) - y) / len(x)
    loss += regularization * np.sum(weights[1:] ** 2) / 2
    gradient[1:] += regularization * weights[1:]
    return float(loss), gradient


def train(rows, *, optimizer='batch', epochs=2500, learning_rate=0.1,
          batch_size=32, regularization=0.0001, seed=42, features=None):
    if optimizer not in ('batch', 'sgd', 'mini-batch'):
        raise ValueError('Unknown optimizer.')
    if epochs < 1 or batch_size < 1:
        raise ValueError('Epochs and batch size must be positive.')
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('Learning rate must be finite and positive.')
    if not math.isfinite(regularization) or regularization < 0:
        raise ValueError('Regularization must be finite and nonnegative.')
    features = FEATURES if features is None else features
    if not features or len(set(features)) != len(features):
        raise ValueError('Features must be a nonempty list without duplicates.')
    if set(row['Hogwarts House'] for row in rows) != set(HOUSES):
        raise ValueError('Training requires examples from all four houses.')
    preprocessing = fit_preprocessing(rows, features)
    x = transform(rows, preprocessing)
    y = np.array([[float(row['Hogwarts House'] == house) for house in HOUSES] for row in rows])
    weights = np.zeros((x.shape[1], len(HOUSES)))
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(epochs):
        if optimizer == 'batch':
            _, gradient = loss_and_gradient(x, y, weights, regularization)
            weights -= learning_rate * gradient
        else:
            order = rng.permutation(len(x))
            size = 1 if optimizer == 'sgd' else batch_size
            # Decay reduces late stochastic oscillations.
            rate = learning_rate / (1 + epoch * 0.02)
            for start in range(0, len(x), size):
                indices = order[start:start + size]
                _, gradient = loss_and_gradient(x[indices], y[indices], weights, regularization)
                weights -= rate * gradient
        if epoch == 0 or (epoch + 1) % 50 == 0 or epoch + 1 == epochs:
            loss, _ = loss_and_gradient(x, y, weights, regularization)
            if not math.isfinite(loss) or not np.all(np.isfinite(weights)):
                raise ValueError('Training diverged; reduce the learning rate.')
            history.append({'epoch': epoch + 1, 'loss': loss})
    return {
        'version': 1, 'classes': list(HOUSES), 'preprocessing': preprocessing,
        'weights': weights.tolist(), 'history': history,
        'training': {'optimizer': optimizer, 'epochs': epochs, 'learning_rate': learning_rate,
                     'batch_size': batch_size, 'regularization': regularization, 'seed': seed},
    }


def predict(rows, model):
    scores = transform(rows, model['preprocessing']) @ np.asarray(model['weights'])
    # Sigmoid is monotonic, so argmax(logit) equals argmax(sigmoid(logit)).
    return [model['classes'][int(index)] for index in np.argmax(scores, axis=1)]


def save_model(model, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as stream:
        json.dump(model, stream, indent=2, allow_nan=False)
        stream.write('\n')


def load_model(path):
    with open(path, encoding='utf-8') as stream:
        model = json.load(stream)
    try:
        if model['version'] != 1 or model['classes'] != HOUSES:
            raise ValueError('Unsupported model version or classes.')
        p = model['preprocessing']
        features = p['features']
        if not features or any(not isinstance(f, str) for f in features) or len(set(features)) != len(features):
            raise ValueError('Invalid model features.')
        n = len(features)
        centers, scales, weights = (np.asarray(a, dtype=float) for a in (p['centers'], p['scales'], model['weights']))
        if centers.shape != (n,) or scales.shape != (n,) or weights.shape != (n + 1, 4):
            raise ValueError('Invalid model array dimensions.')
        if not all(np.all(np.isfinite(a)) for a in (centers, scales, weights)) or np.any(scales <= 0):
            raise ValueError('Invalid model numeric values.')
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError('Malformed model file.') from error
    return model


def stratified_split(rows, fraction=0.2, seed=42):
    if not 0 < fraction < 1:
        raise ValueError('Validation fraction must be between zero and one.')
    rng = np.random.default_rng(seed)
    training, validation = [], []
    for house in HOUSES:
        group = [row for row in rows if row['Hogwarts House'] == house]
        if len(group) < 2:
            raise ValueError(f'Need at least two examples of {house} for validation.')
        order = rng.permutation(len(group))
        count = max(1, min(len(group) - 1, round(len(group) * fraction)))
        validation.extend(group[i] for i in order[:count])
        training.extend(group[i] for i in order[count:])
    return training, validation


def evaluate(rows, model):
    predictions = predict(rows, model)
    confusion = [[0 for _ in HOUSES] for _ in HOUSES]
    correct = 0
    for row, predicted in zip(rows, predictions):
        actual = row['Hogwarts House']
        correct += actual == predicted
        confusion[HOUSES.index(actual)][HOUSES.index(predicted)] += 1
    return {'accuracy': correct / len(rows), 'correct': correct, 'total': len(rows),
            'classes': list(HOUSES), 'confusion_matrix': confusion}
