"""LLMGrader — vLLM-backed judge that verifies mapper output.

Mirrors the structure of ``mapping/llm.py::LLMMapper`` (lazy vLLM
init, batched ``chat`` call, guided JSON decoding) but with one
asset-class-agnostic prompt and the ``GraderResult`` schema.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from djnw import runtime as compat  # noqa: F401 — compat shim on import

from macronews.config.paths import ROOT
from macronews.mapping.grading.schemas import GraderResult

logger = logging.getLogger(__name__)

GRADER_PROMPT_PATH = ROOT / "src" / "macronews" / "mapping" / "grading" / "prompts" / "grader.txt"


@dataclass
class GraderInput:
    """One (article, mapping) judgment task.

    Holds everything the grader sees on a single call. The mapper's
    relevance_score IS shown to the grader: under the verification
    framing, the grader is auditing the mapper's full work product
    (claim + score + evidence), not making an independent first-pass
    judgment.
    """

    article_id: str
    headline: str
    paragraphs: list[str]
    group_name: str
    asset_class: str
    member_names: list[str]
    mapper_score: float
    mapper_evidence_paragraphs: list[int]


class LLMGrader:
    def __init__(
        self,
        model_path: str,
        max_model_len: int = 8192,
        tensor_parallel_size: int = 1,
        prompt_path: Path | str = GRADER_PROMPT_PATH,
    ):
        self.model_path = model_path
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        self.system_prompt = Path(prompt_path).read_text()
        self._llm = None

    # ------------------------------------------------------------------
    # vLLM backend
    # ------------------------------------------------------------------

    def _init_llm(self):
        if self._llm is None:
            from vllm import LLM
            from vllm.config.attention import AttentionConfig
            logger.info(
                "Initializing vLLM grader: %s (tp=%d, max_model_len=%d)",
                self.model_path, self.tensor_parallel_size, self.max_model_len,
            )
            # Pin TRITON_ATTN: vLLM 0.19.0 + VLLM_BATCH_INVARIANT=1 rejects
            # auto-picked None backend on Qwen2-arch. FLASH_ATTN's FA2 kernel
            # is sm_90-only and crashes on B200 (sm_100) with
            # cudaErrorUnsupportedPtxVersion. Triton JIT-compiles for the
            # actual SM, which is what the Gemma 4 mapper auto-selects.
            self._llm = LLM(
                model=self.model_path,
                max_model_len=self.max_model_len,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=0.95,
                attention_config=AttentionConfig(backend="TRITON_ATTN"),
            )

    def grade_batch(
        self,
        items: list[GraderInput],
        max_tokens: int = 1024,
    ) -> list[GraderResult]:
        """Grade a batch of (article, mapping) pairs.

        Returns one ``GraderResult`` per input, in the same order.
        Failures fall through to a default ``GraderResult()`` (verdict =
        incorrect, score = 0, empty evidence) with a logged warning.
        """
        if not items:
            return []

        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        self._init_llm()

        schema = GraderResult.model_json_schema()
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
        logger.info("[vLLM] Grading %d mappings...", len(conversations))
        outputs = self._llm.chat(conversations, sampling_params)

        results: list[GraderResult] = []
        for i, output in enumerate(outputs):
            raw = output.outputs[0].text
            results.append(self._parse_one(raw, i))
        return results

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    @staticmethod
    def build_user_message(item: GraderInput) -> str:
        """Compose the per-mapping user message.

        Layout: [HEADLINE] / [ARTICLE]...[/ARTICLE] / [MAPPER_CLAIM] /
        [MAPPER_EVIDENCE]. Paragraph indices in the article match the
        original mapper input so any reference to indices is comparable
        across mapper and grader.
        """
        article_block = "\n\n".join(
            f"[{i}] {p}" for i, p in enumerate(item.paragraphs)
        )
        members = ", ".join(item.member_names) if item.member_names else "(no members listed)"
        if item.mapper_evidence_paragraphs:
            evidence = ", ".join(str(i) for i in item.mapper_evidence_paragraphs)
        else:
            evidence = "(none — mapper tagged on overall topic alone)"
        return (
            f"[HEADLINE] {item.headline}\n\n"
            f"[ARTICLE]\n{article_block}\n[/ARTICLE]\n\n"
            f"[MAPPER_CLAIM]\n"
            f"Asset group: {item.group_name}\n"
            f"Asset class: {item.asset_class}\n"
            f"Members: {members}\n"
            f"Mapper relevance score: {item.mapper_score:.2f}\n"
            f"[/MAPPER_CLAIM]\n\n"
            f"[MAPPER_EVIDENCE]\n{evidence}\n[/MAPPER_EVIDENCE]"
        )

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------

    def _parse_one(self, raw: str, idx: int) -> GraderResult:
        try:
            return GraderResult.model_validate_json(raw)
        except Exception:
            json_str = self._extract_json(raw)
            if json_str:
                try:
                    return GraderResult.model_validate_json(json_str)
                except Exception:
                    pass
            logger.warning("Failed to parse grader output %d: %s", idx, raw[:200])
            return GraderResult()

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
