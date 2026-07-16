"""A SKIP in the corpus-fingerprint guard is a failure at that gate, not a pass.

test_corpus_fingerprint.py and test_tabular_sidecar_guard.py exist to catch a
corpus-changing bug in loaders.py's tabular-sidecar path. Both SKIP when the model
tokenizer isn't on disk -- which happens on Yale's scratch filesystem by a routine
atime purge (it happened 2026-06-23), not by anyone's choice. A SKIP there means the
guard did not run, and `N passed, M skipped` reads as green -- so escalate it to a
FAILURE unless the user explicitly opts out with MACRONEWS_ALLOW_SKIP=1.
"""
import os

import pytest

_GUARDED_FILES = {"test_corpus_fingerprint.py", "test_tabular_sidecar_guard.py"}
_ALLOW_SKIP = os.environ.get("MACRONEWS_ALLOW_SKIP") == "1"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if _ALLOW_SKIP or not report.skipped or item.path.name not in _GUARDED_FILES:
        return
    if call.excinfo is None or not call.excinfo.errisinstance(pytest.skip.Exception):
        return
    report.outcome = "failed"
    report.longrepr = (
        f"{report.longrepr}\n\nSKIP in {item.path.name} hides the corpus-fingerprint "
        f"guard instead of passing it (e.g. weights purged from scratch by atime). "
        f"Set MACRONEWS_ALLOW_SKIP=1 to accept this skip explicitly."
    )
