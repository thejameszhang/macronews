"""The mapper covers 39 of the 50 groups. This pins the boundary in BOTH
directions, because both halves are load-bearing.

The mapper must not see US equity sectors: GICS membership is a fact about a
firm's revenue split, adjudicated from annual reports, and it is not in the
article text. See docs/superpowers/specs/2026-07-16-sector-exclusion-design.md

group_universe.yaml must KEEP them: it is the shared universe definition with a
separate futures project, and the KG lane and the grader both read all 50.
"""
from pathlib import Path

from macronews.utils.groups import load_group_universe, load_mapper_group_universe

SRC = Path(__file__).resolve().parents[1] / "src" / "macronews"


def test_mapper_universe_is_the_full_universe_minus_the_sector_classes():
    full = load_group_universe()
    mapper = load_mapper_group_universe()

    # The literal string, NOT MAPPER_EXCLUDED_ASSET_CLASSES. Deriving `sectors`
    # from the same constant under test makes both sides of every assertion move
    # together when the constant is wrong -- the guard would then be measuring the
    # constant against itself.
    sectors = {gk for gk, gv in full.items()
               if gv["asset_class"] == "US equity sector"}

    assert len(full) == 50, "group_universe.yaml must still describe all 50 groups"
    assert len(sectors) == 11
    assert len(mapper) == 39
    assert set(mapper) | sectors == set(full), "the filter dropped something else"
    assert set(mapper) & sectors == set(), "a sector group reached the mapper"


def test_the_full_universe_still_carries_every_sector():
    """The YAML data is shared with the futures project. Dropping a group from the
    FILE (rather than from the mapper's view) is the failure this catches."""
    full = load_group_universe()
    classes = {gv["asset_class"] for gv in full.values()}
    assert "US equity sector" in classes
    assert len(classes) == 6

    mapper_classes = {gv["asset_class"] for gv in load_mapper_group_universe().values()}
    assert "US equity sector" not in mapper_classes
    assert len(mapper_classes) == 5


def test_the_mapper_loader_is_not_used_outside_the_mapper():
    """The exclusion is mapper-scoped. Wiring this loader into the KG costs the graph
    its only industry-level nodes; wiring it into the grader makes re-grading an
    existing sector-bearing artifact silently skip every sector row and still exit 0.

    Reach of this guard: it is a source-text scan, so a docstring in kg/ that merely
    NAMES the loader false-fails (loud, trivially fixed), and a dynamic reference
    (getattr(mod, "load_" + "mapper_group_universe")) false-passes. Both judged
    acceptable; an import-graph check buys little here. tests/test_layering.py
    already forbids kg/ importing macronews.pipeline or macronews.mapping at all.
    """
    offenders = []
    for d in (SRC / "kg", SRC / "mapping" / "grading"):
        for p in d.rglob("*.py"):
            if "load_mapper_group_universe" in p.read_text():
                offenders.append(str(p.relative_to(SRC)))
    assert offenders == [], (
        f"mapper-scoped loader used outside the mapper lane: {offenders}"
    )


def test_the_live_mapper_defaults_are_wired_to_39():
    """AC4, asserted DIRECTLY rather than left to emerge.

    Without this, "the mapper covers 39" holds only via a chain of behavioural tests
    that each need their output read at the right moment. An implementer who batches
    the suite to the end could re-pin the corpus fingerprint to the values already on
    main and see nothing go red. These two lines make the invariant self-evident.
    """
    import macronews.pipeline as pipeline
    from macronews.mapping.gate import compile_gate

    assert len(pipeline.ALL_GROUPS) == 39
    assert len(compile_gate()) == 39, (
        "compile_gate's bare default is not the mapper universe -- gate_fires() calls "
        "it with no argument, so the gate and the pipeline have silently diverged"
    )


def test_the_mapper_registers_a_prompt_for_exactly_its_own_classes():
    """llm.py's arity check is symmetric: an unregistered class AND an orphaned
    prompt both raise. It must compare against the MAPPER's classes (5), not the
    file's (6), or deleting equity_sectors.txt stops the mapper from starting."""
    from macronews.mapping.llm import ASSET_CLASS_PROMPT_FILES

    mapper_classes = {gv["asset_class"] for gv in load_mapper_group_universe().values()}
    assert set(ASSET_CLASS_PROMPT_FILES) == mapper_classes
    assert "US equity sector" not in ASSET_CLASS_PROMPT_FILES
    assert not (SRC / "mapping" / "prompts" / "asset_class" / "equity_sectors.txt").exists()


def test_the_real_arity_check_runs_and_returns_the_mappers_five_classes():
    """AC5. The ONLY test in the suite that calls _load_asset_class_rules() for real.

    Every other test drives run_pipeline with a StubMapper whose asset_class_rules()
    is a hardcoded stub, so nothing ever constructs the real rules. That leaves the
    llm.py wiring with no regression cover at all. The full partial-miss matrix, each
    verified by breaking that edit in isolation and running this test:

        miss the local `load_mapper_group_universe` import inside
        `_load_asset_class_rules` only
            -> NameError: name 'load_mapper_group_universe' is not defined

        miss the `group_classes = {...}` arity source only
            -> NameError: name 'load_group_universe' is not defined

        miss both of the above
            -> ValueError: group_universe.yaml contains asset classes with no
               class-specific prompt: ['US equity sector'] (the arity source still
               reads the 50-group universe; prompt is gone)

        miss the `ASSET_CLASS_PROMPT_FILES` sector registration only
            -> FileNotFoundError: Asset-class prompt file missing:
               .../asset_class/equity_sectors.txt (for asset_class='US equity
               sector'). NOT a ValueError: the file-existence loop over
               `ASSET_CLASS_PROMPT_FILES` runs BEFORE the arity check, so a
               registration pointing at a deleted file raises there first.

        apply the import and the arity source only (registration AND file both kept)
            -> ValueError: ASSET_CLASS_PROMPT_FILES declares prompts for classes not
               in group_universe.yaml: ['US equity sector']

    In every case LLMMapper.__init__ never completes and the mapper cannot start --
    while `pytest -q -rs` reports green, 0 skipped, which AC11 treats as done.

    The sibling test above cannot cover this: it recomputes mapper_classes itself,
    so it stays green while the real function -- which does its own lookup inside --
    raises.

    __new__ skips __init__ (which would load a model). _load_asset_class_rules
    touches no self attributes, only module-level constants, so this is CPU-only
    and takes milliseconds.
    """
    from macronews.mapping.llm import LLMMapper

    rules = LLMMapper.__new__(LLMMapper)._load_asset_class_rules()
    assert set(rules) == {"commodity", "currency", "equity index", "rates", "volatility"}
    assert "US equity sector" not in rules
