# DSLR — Data Science × Logistic Regression

A Hogwarts house classifier based on students' course scores. The mandatory programs are implemented, plus bonuses: extra statistics fields, stochastic gradient descent and mini-batch gradient descent.

## Installation and quick start

Python 3.10 or newer. Run the commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python3 logreg_train.py datasets/dataset_train.csv
python3 logreg_predict.py datasets/dataset_test.csv weights.json
python3 -m unittest discover -s tests -v
```

Statistics, plots and bonus optimizers are run separately with the commands below. Ready-made `weights.json`, `houses.csv` and plots in `plots/` are already included. Running a command again overwrites the corresponding result.

## Mandatory programs

### 1. Statistics

```bash
python3 analysis/describe.py datasets/dataset_train.csv
python3 analysis/describe.py datasets/dataset_train.csv --bonus
```

`Count`, `Mean`, `Std`, `Min`, `25%`, `50%`, `75%`, `Max` are calculated by hand. The table is split into groups of four columns so long names fit in the terminal.

- Empty values and non-numeric special values NaN/Inf are left out of the statistics.
- Columns containing text are not treated as numeric.
- The `Index` column is a row number, not a feature, so it is not shown.
- A completely empty column is shown with Count=0 and N/A; its type cannot be determined from a single CSV.
- For a single observation the standard deviation is N/A.
- Std is the sample standard deviation, with denominator `n - 1`.
- Quartiles: sort, position `(n - 1) * q`, linear interpolation between neighbouring values.
- Bonuses: sample Variance, Range and interquartile range IQR.

No library functions are used to compute these values. Values are sorted with our own merge sort, and `math.sqrt` only takes the square root. `max` in the printing function picks the cell width, not the statistical maximum of the data.

### 2. Visualization

```bash
python3 visualization/histogram.py datasets/dataset_train.csv
python3 visualization/scatter_plot.py datasets/dataset_train.csv
python3 visualization/pair_plot.py datasets/dataset_train.csv
```

Each script saves a PNG to `plots/` and opens a window. Add `--no-show` to run without a window. The output path is set with `--output`.

```bash
python3 visualization/histogram.py --course 'Care of Magical Creatures' --no-show --output plots/homogeneous.png
python3 visualization/scatter_plot.py --x Astronomy --y 'Defense Against the Dark Arts' --no-show
python3 visualization/pair_plot.py --selected --no-show --output plots/selected_pairs.png
```

**Answers to the subject's questions for the provided data:**

1. **Histogram:** in `Care of Magical Creatures` the score distributions of all houses overlap strongly. `Arithmancy` is another suitable example. The bins are shared and the plot shows density rather than student counts, because the houses have different sizes.
2. **Scatter plot:** `Astronomy` and `Defense Against the Dark Arts` carry the same information with opposite signs: `Astronomy ≈ -100 × Defense Against the Dark Arts`. Similar features do not have to share a scale or be positively related.
3. **Pair plot:** colours show which courses separate the houses. For example, Charms separates Ravenclaw from Hufflepuff well, Flying helps single out Gryffindor, and Divination singles out Slytherin. 10 courses are kept for training: Astronomy, Herbology, Divination, Muggle Studies, Ancient Runes, History of Magic, Transfiguration, Potions, Charms, Flying.

Arithmancy and Care of Magical Creatures are excluded as weak separators, and Defense Against the Dark Arts as a duplicate of Astronomy. Index, first name, last name, birthday and best hand are not used. Hogwarts House is the target label, not an input feature. This choice is not claimed to be the only or the optimal one.

### 3. Training

```bash
python3 logreg_train.py datasets/dataset_train.csv
```

Creates `weights.json`. Defaults: batch gradient descent, 2500 epochs, learning rate 0.1, L2 regularization 0.0001. Regularization can be turned off with `--regularization 0`.

```bash
python3 logreg_train.py datasets/dataset_train.csv --epochs 3000 --learning-rate 0.1 --output another_model.json
```

For each selected feature the program computes the mean and standard deviation from the training observations only. Missing values are filled with the training mean, then `z = (x - mean) / std` is applied. Constant features get a scale of 1. A training feature with no values at all raises a clear error.

The model file stores the feature names and order, means, scales, weights, house order, training settings and loss history. This way a separate prediction run applies the same preprocessing and never refits on the test data.

### 4. Prediction

```bash
python3 logreg_predict.py datasets/dataset_test.csv weights.json
```

Creates `houses.csv` in exactly this format:

```csv
Index,Hogwarts House
0,Gryffindor
1,Hufflepuff
```

This is an example of the format, not a promise of specific first answers. The original row order and Index values are kept. House labels are not needed in the input CSV for prediction. All required feature columns must be present, but individual values may be empty.

## Bonus optimizers

```bash
python3 logreg_train.py datasets/dataset_train.csv --optimizer sgd --epochs 100 --output weights_sgd.json
python3 logreg_train.py datasets/dataset_train.csv --optimizer mini-batch --batch-size 32 --epochs 250 --output weights_minibatch.json
```

| Method | When weights are updated | Notes |
|---|---|---|
| batch | After the whole training set | Stable gradient, one step per epoch |
| sgd | After every student | More oscillation, many cheap steps |
| mini-batch | After a group, 32 students by default | A compromise between the two |

For SGD and mini-batch the rows are shuffled every epoch with a reproducible `--seed` (42 by default). The step size decays as `learning_rate / (1 + 0.02 * epoch)`. An epoch is one pass over the whole training set, so the same number of epochs means a different number of updates for different methods.

## Accuracy check

`validate.py` checks accuracy the same way the grader does at the defense: it compares a predictions file with a file of true houses by `Index`, using `accuracy_score` from scikit-learn. The model itself does not use sklearn; it is only needed for this check.

At the defense, when `dataset_truth.csv` is available:

```bash
python3 validate.py houses.csv dataset_truth.csv
```

Right now the true houses are only in `dataset_train.csv`, so the local check uses it:

```bash
python3 logreg_predict.py datasets/dataset_train.csv weights.json --output predictions_train.csv
python3 validate.py predictions_train.csv datasets/dataset_train.csv
```

Actual results on `dataset_train.csv`:

| Optimizer | Epochs | Correct | Accuracy |
|---|---:|---:|---:|
| Batch GD | 2500 | 1571 / 1600 | 98.19% |
| Mini-batch GD | 250 | 1571 / 1600 | 98.19% |
| SGD | 100 | 1571 / 1600 | 98.19% |

This check uses the same students the model was trained on. All three methods are wrong on the same 29 students: by their scores they look like students of another house, so no model can guess them from this data. `dataset_test.csv` has no houses, so the real test accuracy can only be measured with `dataset_truth.csv`.

## How the math works

Each student has a vector of normalized scores `x`. A leading 1 is added to it for the intercept (bias). Each house has its own weight vector `w`.

```text
score = x @ w
p = sigmoid(score) = 1 / (1 + exp(-score))
```

One-vs-rest means four binary problems: Gryffindor vs the rest, Hufflepuff vs the rest, and so on. During training the correct house gets y=1 and the others y=0. All four models are computed at once with matrix operations, but their loss functions are independent.

```text
loss = sum_over_classes(mean_over_students(log(1 + exp(score)) - y * score))
gradient = X.T @ (sigmoid(X @ W) - Y) / number_of_students
W = W - learning_rate * gradient
```

`np.logaddexp(0, score)` computes the logarithm stably even for large scores. The sigmoid is written as `exp(-logaddexp(0, -score))` to avoid exponent overflow.

With L2 enabled, `lambda / 2 * sum(W_without_bias ** 2)` is added to the loss and `lambda * W_without_bias` to the gradient. The bias is not penalized.

For prediction the highest of the four scores is chosen. The sigmoid is monotonic, so this is the same house as choosing the highest p. The four one-vs-rest probabilities do not have to sum to 1; this is not softmax.

NumPy is used for arrays, matrix multiplication and element-wise operations. There are no ready-made classifiers, optimizers, `pandas.describe`, or library scaling and data splitting. Matplotlib only draws the plots.

## Code map and reading order

```text
model.py                 # Regression math and data preparation
logreg_train.py          # Training and saving the model
logreg_predict.py        # Loading the model and predicting
validate.py              # Accuracy check
common/
  data.py                # CSV reading and the feature list
  statistics.py          # Shared hand-written math
analysis/
  describe.py            # Statistics table for humans
visualization/
  plotting.py            # Shared plot settings
  histogram.py           # Score distributions
  scatter_plot.py        # Relation between two courses
  pair_plot.py           # All pairs of courses
datasets/                # Source CSV files
plots/                   # Saved images
tests/                   # Automated tests
```

The main regression lives in the root, without wrapper files. No Makefile is needed:
every command runs through Python. The commands above are run from the project root.
You can also change into a folder and pass a relative path to the data, for example:

```bash
cd analysis
python3 describe.py ../datasets/dataset_train.csv
```

When running a plot from its folder, also give the output path:

```bash
cd visualization
python3 histogram.py ../datasets/dataset_train.csv --output ../plots/histogram.png
```

These `cd` examples are independent and start from the project root.
Input and output paths are relative to the terminal's current folder.
The plot scripts and describe add the project root to the import path when run
directly, so they find the shared functions without installing a package.

The plots and the describe table are for humans exploring the data and are required
for submission. Training and prediction do not import `analysis` or `visualization`.
The math functions in `common/statistics.py` are also used when preparing the
training data to compute means and scales. Prediction uses the saved means and scales.

For studying together: one person learns `common`, `analysis` and
`visualization`; the other learns the regression in the root. Then swap and explain
each part to each other: at the defense both of you need to understand the whole project.

Reading order for `model.py`: `fit_preprocessing` → `transform` → `sigmoid` →
`loss_and_gradient` → `train` → `predict`.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

The tests cover known statistics values, empty and single observations, CSV errors, the gradient against numerical finite differences, sigmoid stability, applying training normalization to new data, the quality of all optimizers, save/load, a corrupted model, and a full CLI run checking all 400 output rows.

Before submitting, make sure new source files and the needed results are added to Git: only the contents of the submitted repository are evaluated. Dependencies and instructions are in `requirements.txt` and this README.
