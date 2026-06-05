"""LLMExtractor — vLLM-backed KG Phase 1 fact extractor.

Mirrors src/mapping/llm.py::LLMMapper (lazy vLLM init, batched chat,
guided JSON decoding) but with one article-agnostic prompt and the
KGArticleResult schema. One LLM.chat() invocation per batch; vLLM
schedules N article conversations internally.
"""

from __future__ import annotations

import json
import logging

import compat  # noqa: F401  — transformers / vLLM compat shim

from config.paths import KG_PROMPTS_DIR
from kg.schemas import KGArticleResult

logger = logging.getLogger(__name__)

EXTRACTOR_PROMPT_PATH = KG_PROMPTS_DIR / "extractor.txt"
ENTITY_TYPES_PATH = KG_PROMPTS_DIR / "entity_types.txt"
RELATION_TYPES_PATH = KG_PROMPTS_DIR / "relation_types.txt"

_ENTITY_PLACEHOLDER = "{{ENTITY_TYPES}}"
_RELATION_PLACEHOLDER = "{{RELATION_TYPES}}"


def render_system_prompt() -> str:
    """Render the extractor system prompt with the entity and relation
    taxonomies substituted in.

    Output is byte-identical across calls so vLLM prefix caching can
    reuse the KV cache for the system prompt across every article.
    Anything that would vary per call must NOT enter this function.
    """
    template = EXTRACTOR_PROMPT_PATH.read_text()
    entity_table = ENTITY_TYPES_PATH.read_text().rstrip()
    relation_table = RELATION_TYPES_PATH.read_text().rstrip()
    return (
        template
        .replace(_ENTITY_PLACEHOLDER, entity_table)
        .replace(_RELATION_PLACEHOLDER, relation_table)
    )


def render_user_message(headline: str, paragraphs: list[str]) -> str:
    """Compose the per-article user message in the same indexed format
    the mapper uses, so paragraph indices the LLM emits in
    `evidence_paragraphs` are unambiguous.
    """
    article_block = "\n\n".join(
        f"[{i}] {p}" for i, p in enumerate(paragraphs)
    )
    return (
        f"[HEADLINE] {headline}\n\n"
        f"[ARTICLE]\n{article_block}\n[/ARTICLE]"
    )


class LLMExtractor:
    """Per-article KG fact extractor.

    Gemma 4 31B on a single B200, batch-invariant attention, TP=1.
    Default attention backend (vLLM auto-picks — same as LLMMapper). No
    AttentionConfig override (that's Qwen2-arch-specific, used by the
    grader). Prefix caching is enabled explicitly so the static system
    prompt is cached after the first call across all subsequent calls.
    """

    def __init__(
        self,
        model_path: str,
        max_model_len: int = 65536,  # production context (matches the runner default)
        tensor_parallel_size: int = 1,
    ):
        self.model_path = model_path
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        self.system_prompt = render_system_prompt()
        self._llm = None

    # ------------------------------------------------------------------
    # vLLM backend
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
                # Static system prompt → cache the system KV across all articles.
                enable_prefix_caching=True,
            )

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract_batch(
        self,
        articles: list[dict],
        max_tokens: int = 2048,
    ) -> list[KGArticleResult]:
        """Extract entities + facts for each article.

        Each input dict has the standard loader shape: at minimum
        `headline: str` and `paragraphs: list[str]`. Other keys are
        ignored (the runner handles `id`, `date`, etc.).

        Returns one KGArticleResult per input, in input order. Parse
        failures yield an empty KGArticleResult with a WARN log rather
        than raising.
        """
        if not articles:
            return []

        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        self._init_llm()

        schema = KGArticleResult.model_json_schema()
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            structured_outputs=StructuredOutputsParams(json=schema),
        )

        conversations = [
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": render_user_message(
                    a["headline"], a["paragraphs"]
                )},
            ]
            for a in articles
        ]
        logger.info("[vLLM] Extracting KG from %d articles...", len(conversations))
        outputs = self._llm.chat(conversations, sampling_params)

        results: list[KGArticleResult] = []
        for i, output in enumerate(outputs):
            raw = output.outputs[0].text
            results.append(self._parse_one(raw, i))
        return results

    # ------------------------------------------------------------------
    # Output parsing — mirrors LLMMapper / LLMGrader
    # ------------------------------------------------------------------

    def _parse_one(self, raw: str, idx: int) -> KGArticleResult:
        try:
            return KGArticleResult.model_validate_json(raw)
        except Exception:
            json_str = self._extract_json(raw)
            if json_str:
                try:
                    return KGArticleResult.model_validate_json(json_str)
                except Exception:
                    pass
            logger.warning("Failed to parse KG output %d: %s", idx, raw[:200])
            return KGArticleResult()

    @staticmethod
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
