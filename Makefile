.PHONY: help install test test-synthetic test-full run clean clean-all

PYTHON := python
PIP    := $(PYTHON) -m pip

help:
	@echo "Available targets:"
	@echo "  install         - create venv and install dependencies"
	@echo "  test            - run all pytest tests"
	@echo "  test-synthetic  - run only synthetic tests (no data needed)"
	@echo "  test-full       - run only full-data tests"
	@echo "  run             - run full pipeline 01..06"
	@echo "  clean           - remove generated .npy, .json, .png"
	@echo "  clean-all       - clean + remove venv and __pycache__"

install:
	$(PYTHON) -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

test:
	$(PYTHON) -m pytest tests/ -v

test-synthetic:
	$(PYTHON) -m pytest tests/test_synthetic.py -v

test-full:
	$(PYTHON) -m pytest tests/test_full.py -v

run:
	$(PYTHON) src/01_load.py
	$(PYTHON) src/02_aggregate.py
	$(PYTHON) src/03_matrices.py
	$(PYTHON) src/04_scenarios.py
	$(PYTHON) src/05_integrate.py
	$(PYTHON) src/06_visualize.py

clean:
	rm -f data/processed/*.npy data/processed/*.json
	rm -f out/*.png
	rm -rf .pytest_cache __pycache__ src/__pycache__ tests/__pycache__

clean-all: clean
	rm -rf .venv