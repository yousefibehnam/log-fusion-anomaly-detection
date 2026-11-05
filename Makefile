PYTHON ?= python3
VENV   ?= .venv
BIN     = $(VENV)/bin

.PHONY: install test lint demo synthetic hdfs all clean

install:            ## create a virtual environment and install the package
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install -q -r requirements.txt
	$(BIN)/pip install -q -e . --no-deps

test:               ## run the unit tests
	$(BIN)/pytest

lint:
	$(BIN)/ruff check src tests
	$(BIN)/ruff format --check src tests

demo: install       ## quick end-to-end run on synthetic data (no download needed)
	$(BIN)/logfusion synthetic --seeds 1-2 --work data/demo --out data/demo/results

synthetic:          ## full synthetic experiment (seeds 1-10) -> results/synthetic
	$(BIN)/logfusion synthetic --seeds 1-10

hdfs:               ## download HDFS_v1 and run the real-data experiment -> results/hdfs
	scripts/download_data.sh
	$(BIN)/logfusion hdfs

all: synthetic hdfs

clean:
	rm -rf data/demo data/synthetic .pytest_cache .ruff_cache
