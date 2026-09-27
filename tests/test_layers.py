"""Each layer imports only the layers under it: the job engine imports nothing
of ours, the reference graph imports only the engine (→ docs/design/core-v1.md,
gate G3).

The crossings that exist today are listed in `ALLOWED`, and each is removed by
the step of the core split that makes it unnecessary. An entry that no longer
crosses fails too, so the list only ever shrinks; it is empty when the split
is done.
"""
from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent / "jobsmith"

#: What each layer may import from `jobsmith`, besides itself.
MAY_IMPORT: dict[str, set[str]] = {
    "engine": set(),
    "artifacts": {"engine"},
    "dag": {"engine", "artifacts"},
}

#: (importing file, imported module) crossings still to be removed.
ALLOWED: set[tuple[str, str]] = set()


def _imported(path: Path) -> set[str]:
    """Every `jobsmith` module the file imports, lazy imports included."""
    package = ["jobsmith", *path.relative_to(PACKAGE).parent.parts]
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = package[: len(package) - node.level + 1] if node.level else []
            module = ".".join(base + ([node.module] if node.module else []))
            found.add(module)
    return {name for name in found if name.startswith("jobsmith.")}


def _crossings() -> set[tuple[str, str]]:
    crossings = set()
    for layer, allowed in MAY_IMPORT.items():
        for path in sorted((PACKAGE / layer).rglob("*.py")):
            for module in _imported(path):
                target = module.split(".")[1]
                if target != layer and target not in allowed:
                    crossings.add((str(path.relative_to(PACKAGE)), module))
    return crossings


def test_no_layer_imports_a_layer_above_it():
    new = _crossings() - ALLOWED
    assert not new, f"imports across layers (→ docs/design/core-v1.md): {sorted(new)}"


def test_every_allowed_crossing_still_exists():
    gone = ALLOWED - _crossings()
    assert not gone, f"remove from ALLOWED, the split no longer needs them: {sorted(gone)}"
