"""Validated run configuration for the LLM stages.

Every parameter a stage runs with is resolved here, checked, and handed over as one
frozen object -- so a run cannot start with a value nobody meant.

This exists because of what the argparse defaults were doing. Each below is a real
default, masked in production because the SLURM launchers overrode it, and silently
wrong for anyone who ran a stage by hand:

  --model            ~/models/gemma-4-31b-it   a path that does not exist, and the
                                               superseded 31B model
  --max-model-len    8192                      production passes 65536, and this
                                               drives the loader's token filter, so
                                               the default silently DROPS 88 of
                                               18,859 articles on the 2014-05c shard
                                               -- a different corpus, no error
  --tensor-parallel  settable                  TP=1 is an invariant, not a knob

`extra="forbid"` matters as much as the validators: a renamed or misspelled key
becomes a loud error instead of a field that silently keeps its default -- for
direct construction, since argparse already rejects an unknown `--flag` before
pydantic ever sees it.
"""

import json
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from djnw import runtime as invariants
from macronews.config.paths import (
    ARTICLES_SAMPLE_DIR, DATA_DIR, DJNW_ARTICLES_DIR, GRADER_MODEL, MAPPER_MODEL,
)

Dataset = Literal["gold", "sports", "wikigaming", "djnw"]


def gate_default(dataset: str) -> bool:
    """Whether the keyword gate runs by default for ``dataset``.

    The gate is a production cost optimization. gold/sports/wikigaming are measurement
    sets, and gating them removes the model from the experiment they exist to run: the
    negative controls test that the *model* rejects non-financial text, and a regex
    rejecting it first proves nothing (a model that tagged everything would pass too).
    They are small enough that gating them saves no meaningful GPU time anyway.
    """
    return dataset == "djnw"

# The prompt headroom is per-stage, and it is not a rounding choice -- it decides which
# articles are dropped as too long. Both stages now reserve 8000, but they arrived there
# independently and against different limits (mapper max_model_len 65,536 / 512 output;
# grader 40,960 / 1024 output), so keep them separate: each has to cover its OWN
# rendered-prompt overhead, which is dominated by the per-paragraph [i] markers the
# raw-text length filter cannot see. See each subclass for its measured evidence.
# Do NOT collapse these into one shared constant.
_MIN_ARTICLE_TOKENS = 1024

# ABSOLUTE, so a stage runs from any cwd. Relative paths here would resolve against
# the caller's directory.
_SAMPLE_DIRS: dict[str, Path] = {
    "gold": ARTICLES_SAMPLE_DIR,
    "sports": DATA_DIR / "sports_news_1994_2000",
    "wikigaming": DATA_DIR / "WikiGaming.jsonl",
    "djnw": DJNW_ARTICLES_DIR,
}


def model_max_context(model: Path) -> int:
    """The context length the weights actually support, from their own config."""
    config_path = model / "config.json"
    try:
        cfg = json.loads(config_path.read_text())
    except json.JSONDecodeError as e:
        # Not a ValueError: json.JSONDecodeError IS one, and pydantic wraps any
        # ValueError raised inside a model_validator into a generic ValidationError
        # -- indistinguishable in kind from a usage mistake like a bad flag. A
        # corrupt/half-downloaded weights file is not that, so this must escape
        # pydantic's wrapping and propagate as itself.
        raise RuntimeError(
            f"{config_path} is not valid JSON ({e}). The weights may be corrupt or "
            f"half-downloaded; re-download with slurm/download_llm.sh."
        ) from e
    inner = cfg.get("text_config", cfg)
    limit = inner.get("max_position_embeddings") or cfg.get("max_position_embeddings")
    if not limit:
        raise ValueError(f"{model}/config.json declares no max_position_embeddings")
    return int(limit)


class _LLMStage(BaseModel):
    """Fields shared by every stage that loads a model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: Path
    max_model_len: int = Field(gt=0)

    # Subclasses set their own. See the note above -- they are NOT the same.
    _prompt_headroom_tokens: ClassVar[int]

    @model_validator(mode="after")
    def _model_exists_and_fits(self):
        if not (self.model / "config.json").is_file():
            raise ValueError(
                f"no model at {self.model} (looked for config.json). Scratch "
                f"auto-purges weights by atime; re-download with slurm/download_llm.sh."
            )
        cap = model_max_context(self.model)
        if self.max_model_len > cap:
            raise ValueError(
                f"max_model_len={self.max_model_len:,} exceeds what {self.model.name} "
                f"supports ({cap:,})."
            )
        return self

    @property
    def max_article_tokens(self) -> int:
        """The corpus filter this config implies: articles longer than this are dropped
        before the model sees them. It follows from max_model_len, which is why a
        serving knob quietly decides which articles are seen.

        Mirrors the stage it configures exactly. Changing it changes the corpus.
        """
        return max(_MIN_ARTICLE_TOKENS,
                   self.max_model_len - self._prompt_headroom_tokens)

    def record(self) -> dict:
        """Provenance for the artifact: settings + the invariants they ran under."""
        return {
            **{k: str(v) if isinstance(v, Path) else v for k, v in self},
            "max_article_tokens": self.max_article_tokens,
            "invariants": invariants.record(),
        }


class MapperConfig(_LLMStage):
    """`macronews mapper run` -- articles to asset groups."""

    # Must exceed the gap between a kept article's RAW-TEXT tokens (what the loader
    # filter counts) and its RENDERED-PROMPT tokens (system prompt + a per-paragraph
    # [i] marker on every paragraph + chat template). That marker overhead scales with
    # PARAGRAPH COUNT, not raw length, so a raw-token cap cannot bound it: a 1,509-para
    # article measured 58,445 raw but rendered to 66,594 (+8,149) and crashed its shard.
    # 8000 covers the observed corpus worst case. The principled fix is a render-length
    # guard that skips over-long prompts instead of relying on this reserve.
    _prompt_headroom_tokens: ClassVar[int] = 8000

    model: Path = MAPPER_MODEL
    # A throughput choice, NOT the model's limit (Gemma 4 supports 262,144).
    # max_model_len sizes the KV cache: 65,536 gives ~25x concurrency; the model's max
    # would give ~6x and a far slower run.
    max_model_len: int = 65_536
    # Off in production. The compute-cost benchmark turns it on so vLLM reports
    # prefix-cache hit rate and prefill/decode throughput; it changes reporting
    # only, never computation.
    report_stats: bool = False

    # NOT required: --input-file IS a djnw shard, so the dataset is inferable.
    # Making it required would break the prod launcher, which passes only
    # --input-file/--output-dir.
    dataset: Dataset | None = None
    sample_dir: Path | None = None
    input_file: Path | None = None

    output_file: Path | None = None
    output_dir: Path | None = None

    max_articles: int | None = None
    start_date: str | None = None
    end_date: str | None = None
    random_seed: int | None = None

    keyword_gate: bool | None = None

    @model_validator(mode="after")
    def _resolve(self):
        # An input_file is a djnw shard by definition.
        if self.dataset is None:
            if self.input_file is None:
                raise ValueError("give --dataset, or --input-file (which implies djnw)")
            object.__setattr__(self, "dataset", "djnw")

        if self.sample_dir is None:
            src = self.input_file.parent if self.input_file else _SAMPLE_DIRS[self.dataset]
            object.__setattr__(self, "sample_dir", src)

        if self.keyword_gate is None:
            object.__setattr__(self, "keyword_gate", gate_default(self.dataset))

        if (self.output_file is None) == (self.output_dir is None):
            raise ValueError("give exactly one of output_file / output_dir")
        if self.input_file is not None and self.dataset != "djnw":
            raise ValueError(f"input_file is a djnw shard; dataset is {self.dataset!r}")
        if self.input_file is not None and not self.input_file.is_file():
            raise ValueError(f"no djnw shard at {self.input_file}")

        # The source must exist. `data/WikiGaming.jsonl` does NOT (only
        # articles_sample/ and sports_news_1994_2000/ are on disk), so without this
        # `--dataset wikigaming` validates clean and dies later inside load_articles
        # -- exactly the "a run cannot start with a value nobody meant" claim this
        # class exists to make. Note gold/sports are DIRS and wikigaming is a FILE,
        # hence .exists() rather than .is_dir().
        if not self.sample_dir.exists():
            raise ValueError(
                f"no data for dataset {self.dataset!r} at {self.sample_dir} "
                f"(it does not exist)"
            )
        return self

    @property
    def output_path(self) -> Path:
        """Where this run writes. A shard's name is derived, not restated."""
        if self.output_file is not None:
            return self.output_file
        name = self.input_file.name
        stem = (name[: -len("_clean.jsonl")] if name.endswith("_clean.jsonl")
                else self.input_file.stem)
        return self.output_dir / f"{stem}.jsonl"


class GraderConfig(_LLMStage):
    """`macronews mapper grade` -- QwQ verifies the mapper's tags."""

    # Same reserve as the mapper, and for the same reason: it must cover the gap
    # between an article's RAW-TEXT tokens (what the loader filter counts) and its
    # RENDERED prompt -- grader system prompt + a per-paragraph [i] marker on every
    # paragraph + the claim block + chat template -- PLUS the 1024-token generation
    # budget, since prompt+output must fit max_model_len. Marker overhead scales with
    # PARAGRAPH COUNT, not raw length: a 936-paragraph article measured 37,239 raw but
    # rendered to 42,759 (+5,520) and crashed its shard at 3000. Measured on 1997-12,
    # 8000 leaves ~15k of slack and drops exactly one pathological article.
    _prompt_headroom_tokens: ClassVar[int] = 8000

    model: Path = GRADER_MODEL
    max_model_len: int = 40_960   # QwQ's own maximum
    max_tokens: int = 1024

    mapper_output: Path
    output: Path
    dataset: Literal["gold", "djnw"]
    sample_dir: Path | None = None
    input_file: Path | None = None

    @model_validator(mode="after")
    def _resolve(self):
        if self.dataset == "djnw" and self.input_file is None:
            raise ValueError("dataset 'djnw' needs input_file (the *_clean.jsonl shard)")
        if self.input_file is not None and not self.input_file.is_file():
            raise ValueError(f"no djnw shard at {self.input_file}")
        if self.sample_dir is None and self.dataset == "gold":
            object.__setattr__(self, "sample_dir", _SAMPLE_DIRS["gold"])
        if self.sample_dir is not None and not self.sample_dir.exists():
            raise ValueError(
                f"no data for dataset {self.dataset!r} at {self.sample_dir} "
                f"(it does not exist)"
            )
        if not self.mapper_output.is_file():
            raise ValueError(f"no mapper output at {self.mapper_output}")
        return self
