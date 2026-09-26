#!/usr/bin/env bash
# Mutation testing on the lines this branch changed, against the targeted
# tests: what "break the fix, watch the test fail" did by hand (#134).
# Called by `make mutate TESTS="tests/test_x.py" [BASE=main]`.
# Runs in a scratch worktree of HEAD (cosmic-ray edits files in place), so
# commit first; prints the counts and every surviving mutant.
set -euo pipefail
: "${TESTS:?TESTS=\"tests/test_x.py [...]\" (the tests that should catch it)}"
base=${BASE:-main}
root=$(git rev-parse --show-toplevel)
bin="$root/.venv/bin"
mapfile -t files < <(git -C "$root" diff --name-only "$base"...HEAD -- 'jobsmith/*.py' 'evals/*.py')
[ ${#files[@]} -gt 0 ] || { echo "mutate: no product change on HEAD since $base"; exit 0; }
tmp=$(mktemp -d)
cleanup() { git -C "$root" worktree remove --force "$tmp/w" >/dev/null 2>&1 || true; rm -rf "$tmp"; }
trap cleanup EXIT
git -C "$root" worktree add -q --detach "$tmp/w" HEAD
cd "$tmp/w"

paths=$(printf '"%s", ' "${files[@]}")
cat > "$tmp/cr.toml" <<TOML
[cosmic-ray]
module-path = [${paths%, }]
timeout = ${TIMEOUT:-60}
excluded-modules = []
test-command = "$bin/python -m pytest -x -q -p no:cacheprovider ${TESTS}"

[cosmic-ray.distributor]
name = "local"

[cosmic-ray.filters.git-filter]
branch = "$(git -C "$root" rev-parse "$base")"
TOML

"$bin/python" -m pytest -x -q -p no:cacheprovider $TESTS >/dev/null 2>&1 \
  || { echo "mutate: $TESTS fail before any mutation"; exit 1; }
"$bin/cosmic-ray" init "$tmp/cr.toml" "$tmp/s.sqlite" >/dev/null
"$bin/cr-filter-git" --config "$tmp/cr.toml" "$tmp/s.sqlite" >/dev/null 2>&1
"$bin/cosmic-ray" exec "$tmp/cr.toml" "$tmp/s.sqlite" >/dev/null 2>&1
echo "mutate: ${files[*]} against $TESTS"
# Mutants on the changed lines only (the git filter skips the rest): the
# counts, then each surviving line once, with the operators that survived.
"$bin/python" - "$tmp/s.sqlite" <<'PY'
import sqlite3, sys
from collections import defaultdict
db = sqlite3.connect(sys.argv[1])
rows = db.execute("""select m.module_path, m.start_pos_row, m.operator_name, r.test_outcome
                     from mutation_specs m join work_results r using (job_id)
                     where r.worker_outcome != 'SKIPPED'""").fetchall()
killed = sum(1 for *_, t in rows if t == "KILLED")
lived = defaultdict(set)
for path, line, op, t in rows:
    if t == "SURVIVED":
        lived[(path, line)].add(op.split("/")[-1])
print(f"  {len(rows)} mutants on changed lines: {killed} killed, "
      f"{sum(1 for *_, t in rows if t == 'SURVIVED')} survived on {len(lived)} lines")
for (path, line), ops in sorted(lived.items())[:20]:
    print(f"  survived: {path}:{line}  {', '.join(sorted(ops))}")
if len(lived) > 20:
    print(f"  ... {len(lived) - 20} more lines")
PY
