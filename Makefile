# agent_oo — everyday commands.  `make` or `make help` lists everything.
#
# Variables you can override:
#   make chat LLM=openai        force a provider (anthropic|openai|fake)
#   make api PORT=9000          API port (default 8000)
#   make test T=router          only tests matching a keyword (pytest -k)

VENV := .venv
PY   := $(VENV)/bin/python
RUFF := $(VENV)/bin/ruff

LLM  ?=
PORT ?= 8000
T    ?=

LLM_FLAG  := $(if $(LLM),--llm=$(LLM),)
TEST_ARGS := $(if $(T),-k $(T),)

.DEFAULT_GOAL := help

.PHONY: help install install-all test lint fix check leak-check chat api \
        chat-banking api-banking demo-banking clean

help: ## List available commands
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

$(VENV):
	uv venv --python 3.12 $(VENV)

install: $(VENV) ## Create the venv and install dev + API deps (fake LLMs work out of the box)
	uv pip install -p $(PY) -e ".[dev,api]"

install-all: $(VENV) ## Same + every provider (anthropic, openai, langchain chat models)
	uv pip install -p $(PY) -e ".[dev,api,anthropic,openai,chat-anthropic,chat-openai]"

test: ## Run the test suite (T=<keyword> to filter, e.g. make test T=router)
	$(PY) -m pytest tests/ -q $(TEST_ARGS)

lint: ## Lint with ruff
	$(RUFF) check .

fix: ## Lint and auto-fix what ruff can
	$(RUFF) check . --fix

leak-check: ## Domain-leakage gate: framework + global agent must contain no banking-specific strings
	@! grep -rin --include="*.py" "banking\|banquier\|votre\|analyste" \
		agent_oo/core agent_oo/jobs agent_oo/chat agent_oo/api agent_oo/app \
		&& echo "leak-check: OK (framework is domain-clean)"

check: lint leak-check test ## Everything CI would run: lint + leakage gate + tests

chat: ## Chat with THE GLOBAL AGENT (auto-detects provider; LLM=anthropic|openai|fake to force)
	$(PY) -m agent_oo $(LLM_FLAG)

api: ## Global agent HTTP API + SSE on http://127.0.0.1:PORT, default 8000 (docs at /docs)
	$(PY) -m agent_oo api $(PORT) $(LLM_FLAG)

chat-banking: ## Banking example REPL
	$(PY) -m agent_oo.examples.banking.chat $(LLM_FLAG)

api-banking: ## Banking example HTTP API
	$(PY) -m agent_oo.examples.banking.api $(PORT) $(LLM_FLAG)

demo-banking: ## Scripted banking demo (fakes, no API key needed)
	$(PY) -m agent_oo.examples.banking.main

clean: ## Remove caches, build junk, and generated job reports
	rm -rf .pytest_cache .ruff_cache dist *.egg-info artifacts
	find . -type d -name __pycache__ -not -path "./$(VENV)/*" -exec rm -rf {} +
