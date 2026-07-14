"""The KG depends on the mapper's OUTPUT, never on the mapper's CODE.

The KG is mapper-primed by design: it needs --mapper-file, and that data
dependency is the architecture. The code dependency is not. kg/ used to import
load_articles from pipeline.py -- the mapper's CLI entrypoint -- which made the
two lanes impossible to reason about separately. Corpus loading belongs in
loaders.py, next to the four loaders it dispatches to.

These tests fail if that creeps back.
"""
import ast
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

# The shared core: corpus loading, the group universe, paths. Both lanes may use it.
CORE = {"loaders", "utils", "config"}
# The mapper lane. kg/ may not import from these.
MAPPER_ONLY = {"pipeline", "mapping"}


def _imported_top_levels(py: Path) -> set[str]:
    tree = ast.parse(py.read_text(), filename=str(py))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_kg_does_not_import_mapper_code():
    offenders = {
        str(py.relative_to(SRC)): sorted(_imported_top_levels(py) & MAPPER_ONLY)
        for py in sorted((SRC / "kg").rglob("*.py"))
        if _imported_top_levels(py) & MAPPER_ONLY
    }
    assert not offenders, (
        f"kg/ imports mapper code: {offenders}. The KG consumes the mapper's OUTPUT "
        f"(--mapper-file), not its modules. Shared helpers belong in {sorted(CORE)}."
    )


def test_the_mapper_does_not_import_kg():
    """The other direction, so the one-way dependency stays one-way."""
    for py in sorted((SRC / "mapping").rglob("*.py")) + [SRC / "pipeline.py", SRC / "loaders.py"]:
        assert "kg" not in _imported_top_levels(py), f"{py.name} imports kg"


def test_load_articles_lives_in_loaders():
    """Corpus loading is core, not mapper. Both lanes get it from the same place."""
    import loaders
    assert callable(loaders.load_articles)
