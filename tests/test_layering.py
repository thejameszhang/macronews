"""The mapper must never import a KG module. KG now lives in its own repo (macro-kg);
this guards against it creeping back in."""
import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "macronews"


def _imports(py: Path) -> set[str]:
    tree = ast.parse(py.read_text(), filename=str(py))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(".".join(a.name.split(".")[:2]) for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(".".join(node.module.split(".")[:2]))
    return names


def test_macronews_has_no_kg_references():
    offenders = {str(py.relative_to(SRC)) for py in SRC.rglob("*.py")
                 if any(m == "macronews.kg" for m in _imports(py))}
    assert not offenders, f"macronews still imports kg: {offenders}"
