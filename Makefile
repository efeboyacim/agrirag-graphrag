.DEFAULT_GOAL := help
SHELL := /bin/sh

.PHONY: help init install up down logs ps seed ingest index eval eval-run eval-report 	test lint format typecheck check clean

help: ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

init: ## Create .env from the template (no-op if it exists)
	@test -f .env || (cp .env.example .env && echo "Created .env - add ANTHROPIC_API_KEY when you reach Phase 2")

install: ## Sync the local dev environment (uv provisions Python 3.12)
	uv sync

up: init ## Start the stack and wait for health
	docker compose up -d --build
	@echo "Neo4j browser : http://localhost:7474"
	@echo "API docs      : http://localhost:8000/docs"

down: ## Stop the stack (volumes preserved)
	docker compose down

logs: ## Tail service logs
	docker compose logs -f

ps: ## Show service status
	docker compose ps

seed: ## Load the knowledge graph from data/seed (safe to re-run)
	uv run agrirag-seed --reset

index: ## Rebuild the LanceDB index (free, offline)
	uv run agrirag-index

ingest: ## Rebuild the index AND re-extract graph provenance (costs LLM calls)
	uv run agrirag-index --extract

eval-run: ## Run the agent over the golden set and cache the results
	uv run python -m evals.runner --refresh

eval-report: ## Score the cached results and regenerate evals/reports/latest.md
	uv run python -m evals.report

eval: ## Enforce the evaluation thresholds against the last report (no LLM calls)
	uv run pytest evals/

test: ## Run the unit suite
	uv run pytest tests/unit

test-all: ## Run every test, including integration (needs the stack up)
	uv run pytest tests evals

lint: ## Lint
	uv run ruff check .

format: ## Format and autofix
	uv run ruff format . && uv run ruff check --fix .

typecheck: ## Type-check
	uv run mypy

check: lint typecheck test ## Everything CI runs

clean: ## Stop the stack and delete volumes (destroys the graph)
	docker compose down -v
