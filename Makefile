.PHONY: test lint smoke install

install:
	pip install -e ".[dev]" -c constraints.txt

test:
	pytest tests/ -q

lint:
	ruff check . && ruff format --check .

smoke:
	safeshift run --matrix configs/matrices/quick_matrix.yaml --executor mock --pattern-only --output results/smoke

format:
	ruff format . && ruff check --fix .
