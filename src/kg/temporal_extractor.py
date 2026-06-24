"""LLMTemporalExtractor — 3-pass vLLM-backed temporal KG extractor.

Pass 1: article → list[RawStatement]  (full article visible)
Pass 2: statement → TemporalValidityRange  (statement-only)
Pass 3: statement → list[RawTriplet]       (statement-only)

All three passes are single batched LLM.chat() calls. Pass-2 and Pass-3
outputs are indexed 1:1 with the flattened statement list so salvage parsers
return empty defaults (never drop) to keep lengths aligned.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

import compat  # noqa: F401  — transformers / vLLM compat shim

from config.paths import KG_PROMPTS_DIR
from kg.temporal_schemas import (
    RawStatement,
    RawStatementList,
    RawTriplet,
    RawTripletList,
    TemporalEvent,
    TemporalValidityRange,
)

logger = logging.getLogger(__name__)

# Read prompt templates + the entity/relation taxonomy blocks ONCE at import.
# render_triplet_user_msg runs per statement, so reading the taxonomy files per
# call would be tens of thousands of NFS reads per run.
_STMT_PROMPT = (KG_PROMPTS_DIR / "statement_extraction.txt").read_text()
_TEMP_PROMPT = (KG_PROMPTS_DIR / "temporal_extraction.txt").read_text()
_TRIP_PROMPT = (KG_PROMPTS_DIR / "triplet_extraction.txt").read_text()
_ENTITY_TYPES_BLOCK = (KG_PROMPTS_DIR / "entity_types.txt").read_text().rstrip()
_RELATION_TYPES_BLOCK = (KG_PROMPTS_DIR / "relation_types.txt").read_text().rstrip()


# ---------------------------------------------------------------------------
# Publication-date helper
# ---------------------------------------------------------------------------

def _pub(a: dict) -> str:
    return a.get("date") or a.get("publication_date") or ""


# ---------------------------------------------------------------------------
# Render helpers (pure — no GPU dependency)
# ---------------------------------------------------------------------------

def render_statement_user_msg(
    headline: str,
    paragraphs: list[str],
    mapper_context: str,
    publication_date: str,
) -> str:
    """Pass-1 user message: full article + mapper priming context."""
    article = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(paragraphs))
    return (
        _STMT_PROMPT
        .replace("{{ publication_date }}", publication_date)
        .replace("{{ relevant_asset_groups_block }}", mapper_context)
        .replace("{{ article_text }}", f"[HEADLINE] {headline}\n\n{article}")
    )


def render_temporal_user_msg(s: RawStatement, publication_date: str) -> str:
    """Pass-2 user message: statement metadata only (no article)."""
    return (
        _TEMP_PROMPT
        .replace("{{ statement }}", s.statement)
        .replace("{{ statement_type }}", s.statement_type)
        .replace("{{ temporal_type }}", s.temporal_type)
        .replace("{{ publication_date }}", publication_date)
    )


def render_triplet_user_msg(s: RawStatement) -> str:
    """Pass-3 user message: statement only, no dates or article context."""
    return (
        _TRIP_PROMPT
        .replace("{{ statement }}", s.statement)
        .replace("{{ entity_types_block }}", _ENTITY_TYPES_BLOCK)
        .replace("{{ relation_types_block }}", _RELATION_TYPES_BLOCK)
    )


# ---------------------------------------------------------------------------
# JSON extraction helper (verbatim from src/kg/llm.py:250-273)
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> str | None:
    """Extract the first balanced JSON object from text that may
    contain a preamble. Mirrors LLMMapper._extract_json /
    LLMGrader._extract_json so behavior is consistent across the
    three vLLM-backed modules.
    """
    try:
        json.loads(text)
        return text
    except (json.JSONDecodeError, ValueError):
        pass
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


# ---------------------------------------------------------------------------
# Salvage parsers — always return a value, never drop
# ---------------------------------------------------------------------------

def parse_statements(raw: str, idx: int) -> RawStatementList:
    try:
        return RawStatementList.model_validate_json(raw)
    except Exception:
        js = _extract_json(raw)
        if js:
            try:
                return RawStatementList.model_validate_json(js)
            except Exception:
                pass
        logger.warning("statement parse failed (%d): %s", idx, raw[:200])
        return RawStatementList()


def parse_validity(raw: str, idx: int) -> TemporalValidityRange:
    try:
        return TemporalValidityRange.model_validate_json(raw)
    except Exception:
        js = _extract_json(raw)
        if js:
            try:
                return TemporalValidityRange.model_validate_json(js)
            except Exception:
                pass
        logger.warning("validity parse failed (%d): %s", idx, raw[:200])
        return TemporalValidityRange()


def parse_triplets(raw: str, idx: int) -> RawTripletList:
    try:
        return RawTripletList.model_validate_json(raw)
    except Exception:
        js = _extract_json(raw)
        if js:
            try:
                return RawTripletList.model_validate_json(js)
            except Exception:
                pass
        logger.warning("triplet parse failed (%d): %s", idx, raw[:200])
        return RawTripletList()


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def assemble_events(
    article_id: str,
    publication_date: str,
    statements: list[RawStatement],
    validities: list[TemporalValidityRange],
    triplet_lists: list[list[RawTriplet]],
) -> list[TemporalEvent]:
    created = datetime.fromisoformat(publication_date) if publication_date else None
    events = []
    for s, vr, trips in zip(statements, validities, triplet_lists):
        events.append(TemporalEvent(
            article_id=article_id,
            statement=s.statement,
            statement_type=s.statement_type,
            temporal_type=s.temporal_type,
            triplets=trips,
            valid_at=vr.valid_at,
            invalid_at=vr.invalid_at,
            created_at=created,
            evidence_paragraphs=s.evidence_paragraphs,
        ))
    return events


# ---------------------------------------------------------------------------
# LLMTemporalExtractor
# ---------------------------------------------------------------------------

class LLMTemporalExtractor:
    """3-pass temporal KG extractor backed by vLLM (Gemma 4 31B on B200).

    Pass 1 sees the full article; Passes 2 and 3 see the statement only.
    All three passes are single batched LLM.chat() calls. Salvage parsers
    guarantee 1:1 alignment between Pass-2/3 outputs and the flat statement
    list (empty defaults, never dropped).
    """

    def __init__(
        self,
        model_path: str,
        max_model_len: int = 65536,
        tensor_parallel_size: int = 1,
    ):
        self.model_path = model_path
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        self._llm = None

    # ------------------------------------------------------------------
    # vLLM backend — verbatim from src/kg/llm.py:146-163
    # ------------------------------------------------------------------

    def _init_llm(self):
        if self._llm is None:
            from vllm import LLM
            logger.info(
                "Initializing vLLM extractor: %s (tp=%d, max_model_len=%d, prefix_cache=on)",
                self.model_path, self.tensor_parallel_size, self.max_model_len,
            )
            self._llm = LLM(
                model=self.model_path,
                max_model_len=self.max_model_len,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=0.95,
                # Mapper's vision-skip: avoids the video position-embedding
                # 4D x 3D matmul that vLLM 0.19.0 batch-invariant rejects on B200.
                limit_mm_per_prompt={"video": 0, "image": 0},
                # Each pass's prompt template is a constant prefix across all of
                # that pass's conversations → cache its KV (no system prompt here).
                enable_prefix_caching=True,
            )

    # ------------------------------------------------------------------
    # Batched chat helper
    # ------------------------------------------------------------------

    def _chat(self, convs: list, wrapper, max_tokens: int):
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        sp = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            structured_outputs=StructuredOutputsParams(json=wrapper.model_json_schema()),
        )
        return self._llm.chat(convs, sp)

    # ------------------------------------------------------------------
    # Public extraction entry point
    # ------------------------------------------------------------------

    def extract_batch(
        self,
        articles: list[dict],
        max_tokens: int = 2048,
    ) -> dict[str, list[TemporalEvent]]:
        """Extract temporal events for each article.

        Each input dict must have: `id`, `headline`, `paragraphs`,
        `mapper_context`, and at least one of `date` / `publication_date`.

        Returns a dict mapping article id → list[TemporalEvent]. Parse
        failures yield empty defaults (WARN logged) so the dict always has
        one entry per input article.
        """
        if not articles:
            return {}

        self._init_llm()

        # Pass 1: full article → statements
        p1 = [
            [{"role": "user", "content": render_statement_user_msg(
                a["headline"], a["paragraphs"], a["mapper_context"], _pub(a)
            )}]
            for a in articles
        ]
        logger.info("[vLLM] Pass 1: %d conversations", len(p1))
        stmt_lists = [
            parse_statements(o.outputs[0].text, i)
            for i, o in enumerate(self._chat(p1, RawStatementList, max_tokens))
        ]

        # Flatten statements; track per-article ranges for reassembly
        flat: list[tuple[dict, RawStatement]] = []
        ranges: list[tuple[int, int]] = []
        for a, sl in zip(articles, stmt_lists):
            start = len(flat)
            flat.extend((a, s) for s in sl.statements)
            ranges.append((start, len(flat)))

        # Pass 2 & 3: statement-only (no article)
        p2 = [
            [{"role": "user", "content": render_temporal_user_msg(s, _pub(a))}]
            for a, s in flat
        ]
        p3 = [
            [{"role": "user", "content": render_triplet_user_msg(s)}]
            for a, s in flat
        ]

        logger.info("[vLLM] Pass 2: %d conversations", len(p2))
        vals = [
            parse_validity(o.outputs[0].text, i)
            for i, o in enumerate(self._chat(p2, TemporalValidityRange, max_tokens))
        ]

        logger.info("[vLLM] Pass 3: %d conversations", len(p3))
        trips = [
            parse_triplets(o.outputs[0].text, i).triplets
            for i, o in enumerate(self._chat(p3, RawTripletList, max_tokens))
        ]

        # Reassemble per article
        out: dict[str, list[TemporalEvent]] = {}
        for a, sl, (start, end) in zip(articles, stmt_lists, ranges):
            out[a["id"]] = assemble_events(
                a["id"], _pub(a),
                sl.statements,
                vals[start:end],
                trips[start:end],
            )
        return out
