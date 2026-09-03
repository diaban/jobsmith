# jobsmith — everyday commands.  `make` or `make help` lists everything.
#
# Variables you can override:
#   make chat LLM=openai        force a provider (anthropic|openai|fake)
#   make chat AGENT=banking     run another agent (see jobsmith/agents/)
#   make serve PORT=9000        daemon port (default 8000)
#   make chat DB=agent.db       persist to SQLite (or a postgres:// DSN)
#   make test T=router          only tests matching a keyword (pytest -k)
#   make worktree B=feat/1-x     isolated checkout + venv for one issue

VENV := .venv
PY   := $(VENV)/bin/python
RUFF := $(VENV)/bin/ruff

LLM   ?=
AGENT ?=
DB    ?=
B     ?=
ARGS ?=
PORT ?= 8000
T    ?=

LLM_FLAG   := $(if $(LLM),--llm=$(LLM),)
# `make eval` must cost nothing by default: the fakes unless a provider is named.
EVAL_LLM   := $(if $(LLM),--llm=$(LLM),--llm=fake)
AGENT_FLAG := $(if $(AGENT),--agent=$(AGENT),)
DB_FLAG    := $(if $(DB),--db=$(DB),)
TEST_ARGS := $(if $(T),-k $(T),)
WT_DIR    := $(subst /,-,$(B))

.DEFAULT_GOAL := help

.PHONY: help install install-all test coverage lint fix check leak-check eval eval-llm \
        worktree worktree-rm \
        serve chat jobs \
        chat-banking serve-banking demo-banking clean

help: ## List available commands
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

$(VENV):
	uv venv --python 3.12 $(VENV)

install: $(VENV) ## Create the venv and install dev + API deps (fake LLMs work out of the box)
	uv pip install -p $(PY) -e ".[dev,api]"

install-all: $(VENV) ## Same + every provider and persistence backend
	uv pip install -p $(PY) -e ".[dev,api,anthropic,openai,chat-anthropic,chat-openai,sqlite,postgres]"

test: ## Run the test suite (T=<keyword> to filter, e.g. make test T=router)
	$(PY) -m pytest tests/ -q $(TEST_ARGS)

coverage: ## Test suite with a per-module coverage report
	$(PY) -m pytest tests/ -q --cov=jobsmith --cov-report=term-missing $(TEST_ARGS)

lint: ## Lint with ruff
	$(RUFF) check .

fix: ## Lint and auto-fix what ruff can
	$(RUFF) check . --fix

leak-check: ## Domain-leakage gate: shared code, the default agent and the eval set must contain no banking-specific strings
	@! grep -rin --include="*.py" "banking\|banquier\|votre\|analyste" \
		jobsmith/core jobsmith/jobs jobsmith/chat jobsmith/api jobsmith/app jobsmith/cli \
		jobsmith/agents/default jobsmith/agents/base.py evals \
		&& echo "leak-check: OK (shared code is domain-clean)"

check: lint leak-check test ## Everything CI would run: lint + leakage gate + tests

eval: ## Score the prompts on the golden set — deterministic tier, no API key (ARGS='--repeat 3')
	$(PY) -m evals $(EVAL_LLM) $(AGENT_FLAG) $(ARGS)

eval-llm: ## Same golden set against a REAL provider: opt-in, costs tokens, never gates CI
	$(PY) -m evals $(LLM_FLAG) $(AGENT_FLAG) $(ARGS)

serve: ## Run the DAEMON: it owns the job engine, so jobs outlive their client
	$(PY) -m jobsmith $(LLM_FLAG) $(AGENT_FLAG) $(DB_FLAG) serve --port $(PORT)

chat: ## Chat with the agent (uses the daemon if one runs, else embedded)
	$(PY) -m jobsmith $(LLM_FLAG) $(AGENT_FLAG) $(DB_FLAG) chat

jobs: ## List jobs (add ARGS='--status running')
	$(PY) -m jobsmith $(LLM_FLAG) $(AGENT_FLAG) $(DB_FLAG) jobs $(ARGS)

chat-banking: ## Banking agent REPL (same shell, different agent)
	$(PY) -m jobsmith $(LLM_FLAG) $(DB_FLAG) --agent banking chat

serve-banking: ## Banking agent daemon (HTTP API + SSE)
	$(PY) -m jobsmith $(LLM_FLAG) $(DB_FLAG) --agent banking serve --port $(PORT)

demo-banking: ## Scripted banking demo (fakes, no API key needed)
	$(PY) -m jobsmith.agents.banking.demo

worktree: ## Isolated checkout + ready venv for one issue: make worktree B=feat/1-grounding
	@test -n "$(B)" || { echo "usage: make worktree B=feat/1-grounding"; exit 1; }
	@git worktree add .claude/worktrees/$(WT_DIR) -b $(B)
	@cd .claude/worktrees/$(WT_DIR) \
		&& uv venv --python 3.12 .venv \
		&& uv pip install -p .venv/bin/python -q -e ".[dev,api]"
	@test -f .env && cp .env .claude/worktrees/$(WT_DIR)/.env \
		&& echo "  .env copied (a fresh worktree has no API key otherwise)" || true
	@echo "  ready: .claude/worktrees/$(WT_DIR)  (branch $(B))"

worktree-rm: ## Remove a worktree and its branch: make worktree-rm B=feat/1-grounding
	@test -n "$(B)" || { echo "usage: make worktree-rm B=feat/1-grounding"; exit 1; }
	@# `gh pr merge --delete-branch` already removes both, so every step is a no-op-safe
	@git worktree remove .claude/worktrees/$(WT_DIR) 2>/dev/null || true
	@git worktree prune
	@git branch -d $(B) 2>/dev/null || true
	@echo "  gone: $(B)"

clean: ## Remove caches, build junk, and generated job reports
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov dist *.egg-info artifacts
	find . -type d -name __pycache__ -not -path "./$(VENV)/*" -exec rm -rf {} +
