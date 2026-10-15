PYTHON ?= python3
TRAIN = datasets/dataset_train.csv
TEST = datasets/dataset_test.csv

.PHONY: all describe plots train predict evaluate bonus test
all: describe plots train predict evaluate

describe:
	$(PYTHON) describe.py $(TRAIN)

plots:
	$(PYTHON) histogram.py $(TRAIN) --no-show
	$(PYTHON) scatter_plot.py $(TRAIN) --no-show
	$(PYTHON) pair_plot.py $(TRAIN) --no-show

train:
	$(PYTHON) logreg_train.py $(TRAIN)

predict:
	$(PYTHON) logreg_predict.py $(TEST) weights.json

evaluate:
	$(PYTHON) evaluate.py $(TRAIN) --output results/validation_batch.json

bonus:
	$(PYTHON) describe.py $(TRAIN) --bonus
	$(PYTHON) evaluate.py $(TRAIN) --optimizer mini-batch --epochs 250 --output results/validation_minibatch.json
	$(PYTHON) evaluate.py $(TRAIN) --optimizer sgd --epochs 100 --output results/validation_sgd.json

test:
	$(PYTHON) -m unittest discover -s tests -v
