.PHONY: help install install-dev setup test lint format clean run docker-up docker-down

# Default target
.DEFAULT_GOAL := help

# Color output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m  # No Color

help:  ## Show this help message
	@echo "$(BLUE)F1 Race Strategy Intelligence Engine - Makefile Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

# =============================================================================
# INSTALLATION
# =============================================================================

install:  ## Install production dependencies
	@echo "$(BLUE)Installing production dependencies...$(NC)"
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "$(GREEN)✓ Installation complete$(NC)"

install-dev:  ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pre-commit install
	@echo "$(GREEN)✓ Development installation complete$(NC)"

# =============================================================================
# PROJECT SETUP
# =============================================================================

setup: install-dev setup-db download-data  ## Full project setup (install + database + data)
	@echo "$(GREEN)✓ Project setup complete!$(NC)"
	@echo "$(YELLOW)Run 'make run' to start the API server$(NC)"

setup-db:  ## Initialize PostgreSQL database with TimescaleDB
	@echo "$(BLUE)Setting up database...$(NC)"
	python scripts/setup_db.py
	@echo "$(GREEN)✓ Database setup complete$(NC)"

download-data:  ## Download historical F1 data (2020-2024)
	@echo "$(BLUE)Downloading historical F1 data...$(NC)"
	python scripts/download_historical_data.py --start-year 2020 --end-year 2024
	@echo "$(GREEN)✓ Data download complete$(NC)"

# =============================================================================
# DEVELOPMENT
# =============================================================================

run:  ## Start FastAPI development server
	@echo "$(BLUE)Starting API server...$(NC)"
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

run-prod:  ## Start FastAPI production server (4 workers)
	@echo "$(BLUE)Starting production API server...$(NC)"
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4

run-frontend:  ## Start React frontend development server
	@echo "$(BLUE)Starting frontend...$(NC)"
	cd frontend && npm install && npm run dev

# =============================================================================
# CODE QUALITY
# =============================================================================

format:  ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	black .
	isort .
	@echo "$(GREEN)✓ Code formatted$(NC)"

lint:  ## Run linting checks (ruff + mypy)
	@echo "$(BLUE)Running linting checks...$(NC)"
	ruff check .
	@echo "$(BLUE)Running type checks...$(NC)"
	mypy app/ data_pipeline/ models/ simulation/ decision_engine/ api/
	@echo "$(GREEN)✓ Linting complete$(NC)"

lint-fix:  ## Auto-fix linting issues
	@echo "$(BLUE)Auto-fixing linting issues...$(NC)"
	ruff check . --fix
	@echo "$(GREEN)✓ Auto-fix complete$(NC)"

# =============================================================================
# TESTING
# =============================================================================

test:  ## Run all tests with coverage
	@echo "$(BLUE)Running tests...$(NC)"
	pytest --cov=app --cov=data_pipeline --cov=models --cov-report=term-missing --cov-report=html
	@echo "$(GREEN)✓ Tests complete$(NC)"
	@echo "$(YELLOW)Coverage report: htmlcov/index.html$(NC)"

test-unit:  ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	pytest tests/unit/ -v

test-integration:  ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(NC)"
	pytest tests/integration/ -v

test-e2e:  ## Run end-to-end tests only
	@echo "$(BLUE)Running e2e tests...$(NC)"
	pytest tests/e2e/ -v

test-fast:  ## Run tests excluding slow tests
	@echo "$(BLUE)Running fast tests...$(NC)"
	pytest -m "not slow"

test-watch:  ## Run tests in watch mode (re-run on file changes)
	@echo "$(BLUE)Running tests in watch mode...$(NC)"
	pytest-watch

coverage:  ## Generate and open HTML coverage report
	pytest --cov=app --cov=data_pipeline --cov=models --cov-report=html
	@echo "$(GREEN)✓ Coverage report generated$(NC)"
	python -m webbrowser htmlcov/index.html

# =============================================================================
# DATABASE
# =============================================================================

db-shell:  ## Open PostgreSQL shell
	psql -U f1_user -d f1_strategy

db-reset:  ## Reset database (WARNING: deletes all data)
	@echo "$(RED)WARNING: This will delete all data!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo "$(BLUE)Resetting database...$(NC)"; \
		dropdb --if-exists f1_strategy; \
		createdb f1_strategy; \
		python scripts/setup_db.py; \
		echo "$(GREEN)✓ Database reset complete$(NC)"; \
	fi

db-migrate:  ## Run database migrations
	@echo "$(BLUE)Running database migrations...$(NC)"
	alembic upgrade head
	@echo "$(GREEN)✓ Migrations complete$(NC)"

db-migrate-create:  ## Create a new database migration
	@read -p "Migration name: " name; \
	alembic revision --autogenerate -m "$$name"

# =============================================================================
# DOCKER
# =============================================================================

docker-build:  ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build -t f1-strategy-api:latest .
	@echo "$(GREEN)✓ Docker image built$(NC)"

docker-up:  ## Start all services with Docker Compose
	@echo "$(BLUE)Starting Docker services...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)✓ Services started$(NC)"
	@echo "$(YELLOW)API: http://localhost:8000$(NC)"
	@echo "$(YELLOW)Grafana: http://localhost:3001$(NC)"

docker-down:  ## Stop all Docker services
	@echo "$(BLUE)Stopping Docker services...$(NC)"
	docker-compose down
	@echo "$(GREEN)✓ Services stopped$(NC)"

docker-logs:  ## View Docker logs
	docker-compose logs -f

docker-ps:  ## Show running Docker containers
	docker-compose ps

docker-clean:  ## Remove Docker containers and volumes
	@echo "$(RED)WARNING: This will delete all Docker data!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker-compose down -v; \
		echo "$(GREEN)✓ Docker cleaned$(NC)"; \
	fi

# =============================================================================
# MODELS
# =============================================================================

train-models:  ## Train all ML models
	@echo "$(BLUE)Training ML models...$(NC)"
	python scripts/train_models.py
	@echo "$(GREEN)✓ Model training complete$(NC)"

train-tire-model:  ## Train tire degradation model only
	python scripts/train_models.py --model tire_degradation

train-laptime-model:  ## Train lap time model only
	python scripts/train_models.py --model lap_time

train-sc-model:  ## Train safety car model only
	python scripts/train_models.py --model safety_car

# =============================================================================
# DATA
# =============================================================================

ingest-live:  ## Start live data ingestion worker
	@echo "$(BLUE)Starting live data ingestion...$(NC)"
	python -m data_pipeline.workers.ingestion_worker

process-historical:  ## Process historical data into features
	@echo "$(BLUE)Processing historical data...$(NC)"
	python scripts/process_historical_data.py

# =============================================================================
# MONITORING
# =============================================================================

logs:  ## View application logs
	tail -f logs/app.log

logs-error:  ## View error logs only
	tail -f logs/app.log | grep ERROR

metrics:  ## Open Prometheus metrics dashboard
	@echo "$(BLUE)Opening Prometheus...$(NC)"
	python -m webbrowser http://localhost:9090

# =============================================================================
# CLEANUP
# =============================================================================

clean:  ## Remove build artifacts, cache, and logs
	@echo "$(BLUE)Cleaning project...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/ .coverage
	@echo "$(GREEN)✓ Project cleaned$(NC)"

clean-data:  ## Remove downloaded and processed data (WARNING: deletes data)
	@echo "$(RED)WARNING: This will delete all data files!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf data/raw/* data/processed/* data/features/*; \
		touch data/raw/.gitkeep data/processed/.gitkeep data/features/.gitkeep; \
		echo "$(GREEN)✓ Data cleaned$(NC)"; \
	fi

clean-models:  ## Remove trained model artifacts
	@echo "$(BLUE)Cleaning model artifacts...$(NC)"
	rm -rf models/registry/* models/cache/*
	touch models/registry/.gitkeep
	@echo "$(GREEN)✓ Models cleaned$(NC)"

clean-all: clean clean-data clean-models  ## Remove all generated files

# =============================================================================
# DOCUMENTATION
# =============================================================================

docs:  ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(NC)"
	mkdocs build
	@echo "$(GREEN)✓ Documentation generated$(NC)"

docs-serve:  ## Serve documentation locally
	@echo "$(BLUE)Serving documentation at http://localhost:8001$(NC)"
	mkdocs serve -a 0.0.0.0:8001

# =============================================================================
# RELEASE
# =============================================================================

version:  ## Show current version
	@python -c "import toml; print(toml.load('pyproject.toml')['project']['version'])"

bump-patch:  ## Bump patch version (0.1.0 -> 0.1.1)
	@echo "$(BLUE)Bumping patch version...$(NC)"
	bump2version patch

bump-minor:  ## Bump minor version (0.1.0 -> 0.2.0)
	@echo "$(BLUE)Bumping minor version...$(NC)"
	bump2version minor

bump-major:  ## Bump major version (0.1.0 -> 1.0.0)
	@echo "$(BLUE)Bumping major version...$(NC)"
	bump2version major

# =============================================================================
# CI/CD
# =============================================================================

ci: lint test  ## Run CI pipeline (lint + test)
	@echo "$(GREEN)✓ CI pipeline complete$(NC)"

pre-commit:  ## Run pre-commit hooks on all files
	pre-commit run --all-files

# =============================================================================
# UTILITIES
# =============================================================================

check-deps:  ## Check for outdated dependencies
	pip list --outdated

update-deps:  ## Update all dependencies (use with caution)
	pip install --upgrade -r requirements.txt

shell:  ## Open Python shell with project context
	python -i -c "from app import *; from data_pipeline.schemas import *"

info:  ## Show project information
	@echo "$(BLUE)Project: F1 Race Strategy Intelligence Engine$(NC)"
	@echo "Version: $$(python -c 'import toml; print(toml.load(\"pyproject.toml\")[\"project\"][\"version\"])')"
	@echo "Python: $$(python --version)"
	@echo "Environment: $$(echo $$ENVIRONMENT)"
