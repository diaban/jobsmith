#!/usr/bin/env bash
# Merge several open PRs onto origin/main in a scratch worktree, then run the
# whole suite and the structural tier on the result. Called by `make combo`.
#   PRS="127 128"
# Prints the conflicting files, or one line per check. Nothing is pushed.
set -uo pipefail
: "${PRS:?PRS=\"<pr> <pr> ...\"}"
root=$(git rev-parse --show-toplevel)
py="$root/.venv/bin/python"
tmp=$(mktemp -d)
cleanup() {
  git -C "$root" worktree remove --force "$tmp/w" >/dev/null 2>&1
  for n in $PRS; do git -C "$root" update-ref -d "refs/combo/$n" 2>/dev/null; done
  rm -rf "$tmp"
}
trap cleanup EXIT

git -C "$root" fetch -q origin main $(for n in $PRS; do echo "+pull/$n/head:refs/combo/$n"; done) || exit 1
git -C "$root" worktree add -q --detach "$tmp/w" origin/main || exit 1
cd "$tmp/w" || exit 1
for n in $PRS; do
  if ! git merge -q --no-edit "refs/combo/$n" >/dev/null 2>&1; then
    echo "combo: #$n conflicts with what precedes it: $(git diff --name-only --diff-filter=U | tr '\n' ' ')"
    exit 1
  fi
done
echo "combo: origin/main + #${PRS// / + #} merged cleanly"

# Run from the scratch tree: `python -m` puts it first on sys.path, so its
# code is what is imported, with the main checkout's venv.
"$py" -m pytest tests/ -q -p no:cacheprovider >"$tmp/tests.log" 2>&1; tests=$?
echo "tests: $(tail -1 "$tmp/tests.log")"
TAVILY_API_KEY= ANTHROPIC_API_KEY= OPENAI_API_KEY= \
  "$py" -m evals --llm fake --no-write >"$tmp/eval.log" 2>&1; evals=$?
echo "structural tier: $(grep -E '^overall' "$tmp/eval.log" | tr -s ' ')"
[ $tests -eq 0 ] && [ $evals -eq 0 ]
