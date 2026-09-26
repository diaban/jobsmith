#!/usr/bin/env bash
# Probe one node on BASE (default: main) and on the working tree, in parallel,
# then print one line per case: before -> after. Called by `make probe`.
#   NODE=document_intent READ=document_formats CASES=evals/probes/document_intent.json
#   optional: GREP=<regex>  N=10  BASE=main  LLM=openai  MAX=300 (calls, both sides)
set -euo pipefail
: "${NODE:?NODE=<graph node>}" "${READ:?READ=<state key>}" "${CASES:?CASES=<json file>}"
root=$(git rev-parse --show-toplevel)
py="$root/.venv/bin/python"
tmp=$(mktemp -d)
cleanup() { git -C "$root" worktree remove --force "$tmp/base" >/dev/null 2>&1 || true; rm -rf "$tmp"; }
trap cleanup EXIT

# The base worktree has no .env: export it once, for both sides.
if [ -f "$root/.env" ]; then set -a; . "$root/.env"; set +a; fi
git -C "$root" worktree add -q --detach "$tmp/base" "${BASE:-main}"

args=(--node "$NODE" --read "$READ" --cases "$(realpath "$CASES")" -n "${N:-10}"
      --max-calls "$(( ${MAX:-300} / 2 ))")
[ -n "${GREP:-}" ] && args+=(--grep "$GREP")
[ -n "${LLM:-}" ] && args+=(--llm "$LLM")

quiet() { grep -v '^\[' >&2 || true; }     # drop the composition banners
(cd "$tmp/base" && "$py" -m evals.probe "${args[@]}" --out "$tmp/before.json" >/dev/null 2> >(quiet)) & before=$!
(cd "$root" && "$py" -m evals.probe "${args[@]}" --out "$tmp/after.json" >/dev/null 2> >(quiet)) & after=$!
wait "$before"; wait "$after"

echo "probe $NODE.$READ — ${BASE:-main} -> working tree"
"$py" -m evals.probe --compare "$tmp/before.json" "$tmp/after.json"
