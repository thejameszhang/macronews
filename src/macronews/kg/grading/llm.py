"""LLMKGGrader — vLLM-backed judge that grades one extracted KG fact at a time.

Mirrors src/macronews/mapping/grading/llm.py (lazy vLLM init, batched chat, guided JSON decode,
parse-with-fallback). Independent of the extractor AND schema-blind: the grader
is never handed the entity/relation taxonomy, so editing the extractor's type
files never changes this ruler. It judges each typed fact from an outside
reader's point of view.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import macronews.compat as compat  # noqa: F401  — transformers / vLLM compat shim

from macronews.kg.grading.schemas import KGStatementVerdict

logger = logging.getLogger(__name__)

GRADER_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "grader.txt"


@dataclass
class KGGraderInput:
    """Everything the judge sees for one (article, statement) call."""

    article_id: str
    headline: str
    paragraphs: list[str]          # FULL paragraph list (re-loaded from source)
    statement: str
    statement_type: str            # FACT / OPINION / PREDICTION
    triplets: list[dict]           # each: subject, subject_type, relation,
                                   #   object, object_type, value
    evidence_paragraphs: list[int]


class LLMKGGrader:
    def __init__(
        self,
        model_path: str,
        max_model_len: int = 40960,
        tensor_parallel_size: int = 1,
        prompt_path: Path | str = GRADER_PROMPT_PATH,
    ):
        self.model_path = model_path
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        # Schema-blind: the grader is NOT handed the entity/relation taxonomy, so
        # editing the extractor's type files never moves this ruler. The system
        # prompt is grader.txt verbatim.
        self.system_prompt = Path(prompt_path).read_text()
        self._llm = None

    def _init_llm(self):
        if self._llm is None:
            from vllm import LLM
            from vllm.config.attention import AttentionConfig
            logger.info(
                "Initializing vLLM KG grader: %s (tp=%d, max_model_len=%d)",
                self.model_path, self.tensor_parallel_size, self.max_model_len,
            )
            # TRITON_ATTN, NOT FLASH_ATTN: FA2 is sm_90-only and crashes on
            # B200/sm_100; vLLM 0.19.0 + VLLM_BATCH_INVARIANT=1 rejects None
            # on Qwen2-arch. (Same pin as src/macronews/mapping/grading/llm.py.)
            self._llm = LLM(
                model=self.model_path,
                max_model_len=self.max_model_len,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=0.95,
                enable_prefix_caching=True,
                attention_config=AttentionConfig(backend="TRITON_ATTN"),
            )

    def grade_batch(
        self,
        items: list[KGGraderInput],
        max_tokens: int = 2048,
    ) -> list[KGStatementVerdict]:
        """Grade a batch of facts; one verdict per input, same order.

        Parse failures fall through to a default KGStatementVerdict() with a logged
        warning (should be ~0 under structured JSON decoding — verify in smoke).
        """
        if not items:
            return []
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        self._init_llm()
        schema = KGStatementVerdict.model_json_schema()
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            structured_outputs=StructuredOutputsParams(json=schema),
        )
        conversations = [
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.build_user_message(item)},
            ]
            for item in items
        ]
        logger.info("[vLLM] Grading %d statements...", len(conversations))
        outputs = self._llm.chat(conversations, sampling_params)
        return [self._parse_one(o.outputs[0].text, i) for i, o in enumerate(outputs)]

    @staticmethod
    def build_user_message(item: KGGraderInput) -> str:
        """Compose: [HEADLINE] / [ARTICLE] / [STATEMENT] / [TRIPLETS].

        The full article sits BEFORE any statement-specific content so that, for
        all statements of one article, the prefix (system + headline + article) is
        identical — vLLM's prefix cache then reuses the article KV across them.
        """
        article_block = "\n\n".join(
            f"[{i}] {p}" for i, p in enumerate(item.paragraphs)
        )
        evidence = (
            ", ".join(str(i) for i in item.evidence_paragraphs)
            if item.evidence_paragraphs else "(none)"
        )
        triplet_lines = []
        for n, t in enumerate(item.triplets, 1):
            val = t.get("value")
            tail = f" = {val}" if val else ""
            triplet_lines.append(
                f"{n}. ({t['subject']} : {t['subject_type']}) "
                f"-- {t['relation']} --> "
                f"({t['object']} : {t['object_type']}){tail}"
            )
        return (
            f"[HEADLINE] {item.headline}\n\n"
            f"[ARTICLE]\n{article_block}\n[/ARTICLE]\n\n"
            f"[STATEMENT] ({item.statement_type}) {item.statement}\n"
            f"Extractor-cited evidence paragraphs: {evidence}\n\n"
            f"[TRIPLETS]\n" + "\n".join(triplet_lines) + "\n[/TRIPLETS]"
        )

    def _parse_one(self, raw: str, idx: int) -> KGStatementVerdict:
        try:
            return KGStatementVerdict.model_validate_json(raw)
        except Exception:
            json_str = self._extract_json(raw)
            if json_str:
                try:
                    return KGStatementVerdict.model_validate_json(json_str)
                except Exception:
                    pass
            logger.warning("Failed to parse KG grader output %d: %s", idx, raw[:200])
            return KGStatementVerdict()

    @staticmethod
    def _extract_json(text: str) -> str | None:
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
