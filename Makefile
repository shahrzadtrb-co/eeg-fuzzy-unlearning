.PHONY: install lint test run sweep demo docker

install:
	pip install -e ".[dev]"

lint:
	ruff check src tests
	ruff format --check src tests

test:
	pytest -q

run:
	eeg-unlearn run --config configs/default.yaml

sweep:
	eeg-unlearn sweep --config configs/default.yaml --seeds 0 1 2 3 4 --output-dir results/sweep

demo:
	eeg-unlearn run --synthetic --output-dir results/demo

docker:
	docker build -t eeg-unlearning .
