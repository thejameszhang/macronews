"""
Publish the assembled per-asset-class mapper system prompt as a data file.

The production pipeline (pipeline.py) never writes the rendered prompt to
disk -- it substitutes the class-specific rules into mapper.txt in memory and
hands the result straight to the LLM. Downstream consumers that need the
exact rendered prompt (e.g. the paper's mapping-example figure) used to
re-derive it by importing macronews code directly. This script performs that
same substitution once per asset class and writes the result to
results/mapping/prod/1996-2025/, so those consumers can read a data file
instead of importing the mapper.

Mirrors the two-placeholder .replace() assembly in pipeline.py's
_run_per_class() / llm.py's LLMMapper._load_asset_class_rules(): mapper.txt
with {{ASSET_CLASS_DISQUALIFIERS}} and {{ASSET_CLASS_POSITIVES}} replaced by
the disqualifier/positive halves of the class's asset_class/*.txt file, split
on the _CLASS_SECTION_SPLIT marker and each stripped of surrounding newlines.

    .venv/bin/python -m macronews.mapping.publish_prompt
"""

from macronews.config.paths import PROMPTS_DIR, RESULTS_DIR
from macronews.mapping.llm import (
    ASSET_CLASS_DISQUALIFIERS_PLACEHOLDER,
    ASSET_CLASS_POSITIVES_PLACEHOLDER,
    ASSET_CLASS_PROMPT_FILES,
    _CLASS_SECTION_SPLIT,
)

OUT_DIR = RESULTS_DIR / "mapping/prod/1996-2025"


def render_prompt(asset_class: str) -> str:
    """Assemble the rendered mapper system prompt for one asset class."""
    template = (PROMPTS_DIR / "mapper.txt").read_text()
    cls_text = (PROMPTS_DIR / "asset_class" / ASSET_CLASS_PROMPT_FILES[asset_class]).read_text()
    parts = cls_text.split(_CLASS_SECTION_SPLIT)
    disqualifiers, positives = (p.strip("\n") for p in parts)
    return template.replace(
        ASSET_CLASS_DISQUALIFIERS_PLACEHOLDER, disqualifiers
    ).replace(
        ASSET_CLASS_POSITIVES_PLACEHOLDER, positives
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for asset_class in ASSET_CLASS_PROMPT_FILES:
        prompt = render_prompt(asset_class)
        slug = asset_class.replace(" ", "_")
        out = OUT_DIR / f"rendered_prompt_{slug}.txt"
        out.write_text(prompt)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
