.PHONY: help install install-dev train inference test clean docker-build docker-run lint format

# Variables
PYTHON := python3
PIP := pip3
DOCKER_IMAGE := ml-project:latest
CONFIG_LOCAL := config/local.yaml
CONFIG_PROD := config/prod.yaml

help:
	@echo "Available commands:"
	@echo "  install        - Install production dependencies"
	@echo "  install-dev    - Install development dependencies"
	@echo "  train          - Train model locally"
	@echo "  train-prod     - Train model with production config"
	@echo "  inference      - Run inference locally"
	@echo "  test           - Run tests"
	@echo "  lint           - Run linters"
	@echo "  format         - Format code"
	@echo "  clean          - Clean temporary files"
	@echo "  docker-build   - Build Docker image"
	@echo "  docker-run     - Run Docker container"
	@echo "  docker-compose - Start services with docker-compose"

install:
	$(PIP) install -r requirements-prod.txt

install-dev:
	$(PIP) install -r requirements-dev.txt

train:
	$(PYTHON) src/entrypoint/train.py --config $(CONFIG_LOCAL)

train-prod:
	$(PYTHON) src/entrypoint/train.py --config $(CONFIG_PROD)

inference:
	$(PYTHON) src/entrypoint/inference.py

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:
	flake8 src/ tests/
	pylint src/ tests/
	mypy src/

format:
	black src/ tests/
	isort src/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf dist
	rm -rf build

docker-build:
	docker build -t $(DOCKER_IMAGE) .

docker-run:
	docker run -p 8000:8000 -v $(PWD)/models:/app/models $(DOCKER_IMAGE)

docker-compose:
	docker-compose up -d

docker-compose-down:
	docker-compose down

setup-dirs:
	mkdir -p data/01-raw data/02-preprocessed data/03-features data/04-predictions
	mkdir -p models logs artifacts notebooks

mlflow:
	mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db