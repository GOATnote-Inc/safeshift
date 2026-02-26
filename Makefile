.PHONY: test lint smoke install

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -q

lint:
	ruff check . && ruff format --check .

smoke:
	safeshift run --matrix configs/matrices/quick_matrix.yaml --executor mock

format:
	ruff format . && ruff check --fix .
