"""The production mapper: one LLM call per (article, asset-group) pair.

The output field order IS the chain of thought -- evidence_paragraphs, then
relevance_score, then relevant. Reordering the schema reorders the reasoning.

The prompt puts the article FIRST and the [ASSET_GROUP] block last, so all 50 groups
of an article share a prefix and vLLM's prefix-cache computes the article once. That
layout is why the keyword gate saves 88% of calls but only 70% of GPU time: skipping
a call does not skip the shared prefill.

No FP8 (weights or KV cache) -- both were tested and both hurt mapping quality.
"""

import json
import logging

from djnw import runtime as compat  # noqa: F401 — compat shim on import

from macronews.config.paths import PROMPTS_DIR
from macronews.mapping.base import BaseMapper
from macronews.mapping.schemas import SingleAssetResult

logger = logging.getLogger(__name__)

# Placeholder default; every stage of the macronews pipeline sets
# mapper.system_prompt explicitly before running, so this value is never
# used at runtime. Kept as a kwarg default for backwards compatibility.
DEFAULT_SYSTEM_PROMPT = ""

# Map each asset_class value in group_universe.yaml to its prompt file under
# PROMPTS_DIR/asset_class/. Unknown classes raise at mapper init time.
# Note: "rates" is a unified class collapsing the prior "government bond"
# and "short-term interest rate" classes — the group structure puts both
# under one curve per issuer.
ASSET_CLASS_PROMPT_FILES = {
    "currency":         "currency.txt",
    "rates":            "rates.txt",
    "commodity":        "commodities.txt",
    "US equity sector": "equity_sectors.txt",
    "equity index":     "equity_indices.txt",
    "volatility":       "volatility_indices.txt",
}

ASSET_CLASS_DISQUALIFIERS_PLACEHOLDER = "{{ASSET_CLASS_DISQUALIFIERS}}"
ASSET_CLASS_POSITIVES_PLACEHOLDER = "{{ASSET_CLASS_POSITIVES}}"

# In-file delimiter between the class-specific NOT-RELEVANT bullets and the
# positive-rule block. Stripped at load time; the LLM never sees this marker.
_CLASS_SECTION_SPLIT = "===POSITIVE RULES==="


class LLMMapper(BaseMapper):
    def __init__(
        self,
        model_path: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_model_len: int = 512,
        tensor_parallel_size: int = 1,
    ):
        self.model_path = model_path
        self.system_prompt_template = system_prompt
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        self._llm = None
        self._asset_list_str = self._get_formatted_asset_list()
        self._asset_class_rules = self._load_asset_class_rules()

    def _load_asset_class_rules(self) -> dict[str, tuple[str, str]]:
        """Load asset-class-specific rule files split into (disqualifiers,
        positives) sections on the ``_CLASS_SECTION_SPLIT`` marker. Verify
        every class in the group universe has a registered prompt file and
        that each file has exactly one split marker."""
        from macronews.utils.groups import load_group_universe

        asset_class_dir = PROMPTS_DIR / "asset_class"
        rules: dict[str, tuple[str, str]] = {}
        for ac, fname in ASSET_CLASS_PROMPT_FILES.items():
            fpath = asset_class_dir / fname
            if not fpath.exists():
                raise FileNotFoundError(
                    f"Asset-class prompt file missing: {fpath} (for asset_class={ac!r})"
                )
            text = fpath.read_text()
            parts = text.split(_CLASS_SECTION_SPLIT)
            if len(parts) != 2:
                raise ValueError(
                    f"{fpath}: expected exactly one {_CLASS_SECTION_SPLIT!r} "
                    f"separator, found {len(parts) - 1}"
                )
            disqualifiers, positives = (p.strip("\n") for p in parts)
            rules[ac] = (disqualifiers, positives)

        group_classes = {gv["asset_class"] for gv in load_group_universe().values()}
        missing_in_prompts = group_classes - rules.keys()
        unused_prompts = rules.keys() - group_classes
        if missing_in_prompts:
            raise ValueError(
                f"group_universe.yaml contains asset classes with no class-specific "
                f"prompt: {sorted(missing_in_prompts)}"
            )
        if unused_prompts:
            raise ValueError(
                f"ASSET_CLASS_PROMPT_FILES declares prompts for classes not in "
                f"group_universe.yaml: {sorted(unused_prompts)}"
            )
        return rules

    def asset_class_rules(self, asset_class: str) -> tuple[str, str]:
        """Return (disqualifiers, positives) for an asset class, or raise."""
        if asset_class not in self._asset_class_rules:
            raise KeyError(
                f"No asset-class-specific rules registered for asset_class={asset_class!r}"
            )
        return self._asset_class_rules[asset_class]

    @property
    def system_prompt(self) -> str:
        """Inject the dynamic asset list into the system prompt template."""
        return self.system_prompt_template.replace("{{ASSET_LIST}}", self._asset_list_str)

    @system_prompt.setter
    def system_prompt(self, value: str):
        self.system_prompt_template = value

    def _get_formatted_asset_list(self) -> str:
        """Build a hierarchical string representation of the full asset universe."""
        from macronews.utils.config import load_asset_universe
        from macronews.config.paths import ASSET_UNIVERSE_YAML
        
        universe = load_asset_universe(ASSET_UNIVERSE_YAML)
        
        # Group by asset_class
        by_class = {}
        for sym, info in universe.items():
            cls = info.get("asset_class", "other")
            by_class.setdefault(cls, []).append((sym, info.get("name", sym)))
            
        lines = []
        
        # 1. GICS Sectors
        if "US equity sector" in by_class:
            lines.append("**EQUITY SECTORS (GICS)**")
            sectors = [f"{sym} ({name})" for sym, name in sorted(by_class["US equity sector"])]
            lines.append(", ".join(sectors))
            lines.append("")

        # 2. Commodities (nested)
        if "commodity" in by_class:
            lines.append("**COMMODITIES**")
            # Sub-group commodities by simple heuristic or known prefixes
            energy = []
            metals = []
            agri = []
            for sym, name in by_class["commodity"]:
                if sym in ["CL", "BRN", "RB", "NG", "HO", "G"]:
                    energy.append(f"{sym} ({name})")
                elif sym in ["GC", "SI", "HG", "AHD", "NID", "PBD", "ZSD"]:
                    metals.append(f"{sym} ({name})")
                else:
                    agri.append(f"{sym} ({name})")
            
            if energy: lines.append(f"- **Energy:** {', '.join(energy)}")
            if metals: lines.append(f"- **Metals:** {', '.join(metals)}")
            if agri:   lines.append(f"- **Agri/Softs/Livestock:** {', '.join(agri)}")
            lines.append("")

        # 3. Bonds
        if "government bond" in by_class:
            lines.append("**GOVERNMENT BONDS**")
            # Sub-group bonds by region
            us = []
            eu = []
            other = []
            for sym, name in by_class["government bond"]:
                if sym in ["ZT", "ZF", "ZN", "TN", "ZB", "UB"]:
                    us.append(f"{sym} ({name})")
                elif sym in ["FGBS", "FGBM", "FGBL", "FGBX", "FOAT", "FBTP"]:
                    eu.append(f"{sym} ({name})")
                else:
                    other.append(f"{sym} ({name})")
            
            if us:    lines.append(f"- **US:** {', '.join(us)}")
            if eu:    lines.append(f"- **Europe:** {', '.join(eu)}")
            if other: lines.append(f"- **Other:** {', '.join(other)}")
            lines.append("")

        # 4. Currencies
        if "currency" in by_class:
            lines.append("**CURRENCIES**")
            curs = [f"{sym} ({name})" for sym, name in sorted(by_class["currency"])]
            lines.append(", ".join(curs))
            lines.append("")

        # 5. Equities (Indices)
        if "equity index" in by_class:
            lines.append("**EQUITY INDEX FUTURES**")
            us = []
            eu = []
            other = []
            for sym, name in by_class["equity index"]:
                if sym in ["ES", "NQ", "RTY", "YM", "EMD"]:
                    us.append(f"{sym} ({name})")
                elif sym in ["FESX", "FDAX", "FCE", "FIB", "FIE", "FTI", "FSMI", "FATX"]:
                    eu.append(f"{sym} ({name})")
                else:
                    other.append(f"{sym} ({name})")
            
            if us:    lines.append(f"- **US:** {', '.join(us)}")
            if eu:    lines.append(f"- **Europe:** {', '.join(eu)}")
            if other: lines.append(f"- **Asia/Other:** {', '.join(other)}")
            lines.append("")

        # 6. Short-Term Interest Rates (STIR)
        if "short-term interest rate" in by_class:
            lines.append("**SHORT-TERM INTEREST RATE FUTURES (STIR)**")
            stirs = [f"{sym} ({name})" for sym, name in sorted(by_class["short-term interest rate"])]
            lines.append(", ".join(stirs))
            lines.append("")

        # 7. Volatility
        if "volatility" in by_class:
            lines.append("**VOLATILITY**")
            vols = [f"{sym} ({name})" for sym, name in sorted(by_class["volatility"])]
            lines.append(", ".join(vols))
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # vLLM backend
    # ------------------------------------------------------------------

    def _init_llm(self):
        if self._llm is None:
            from vllm import LLM
            logger.info(f"Initializing vLLM with model: {self.model_path} (tp={self.tensor_parallel_size})")
            self._llm = LLM(
                model=self.model_path,
                max_model_len=self.max_model_len,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=0.95,
                # Gemma 4 is multimodal but we only send text. Skipping vision
                # warmup avoids the video position-embedding 4D x 3D matmul that
                # vLLM 0.19.0's batch-invariant op can't handle on B200.
                limit_mm_per_prompt={"video": 0, "image": 0},
            )

    def map_single_asset(
        self,
        texts: list[str],
        max_tokens: int = 512,
    ) -> list[SingleAssetResult]:
        """Per-asset mapping: one call per (article/paragraph, asset) pair.

        Uses guided decoding with SingleAssetResult schema.
        """
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        self._init_llm()

        schema = SingleAssetResult.model_json_schema()
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            structured_outputs=StructuredOutputsParams(json=schema),
        )

        conversations = [self._build_conversation(t) for t in texts]
        logger.info(f"[vLLM] Single-asset mapping {len(conversations):,} pairs...")
        outputs = self._llm.chat(conversations, sampling_params)

        results = []
        for i, output in enumerate(outputs):
            raw = output.outputs[0].text
            logger.info("[vLLM] SingleAsset %d (len=%d): %s", i, len(raw), raw[:500])
            try:
                result = SingleAssetResult.model_validate_json(raw)
            except Exception:
                json_str = self._extract_json(raw)
                if json_str:
                    try:
                        result = SingleAssetResult.model_validate_json(json_str)
                    except Exception:
                        logger.warning("Failed to parse single-asset output %d: %s", i, raw[:200])
                        result = SingleAssetResult(relevant=False)
                else:
                    logger.warning("No JSON in single-asset output %d: %s", i, raw[:200])
                    result = SingleAssetResult(relevant=False)
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Shared
    # ------------------------------------------------------------------

    def _build_conversation(self, headline: str) -> list[dict]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": headline},
        ]

    @staticmethod
    def _extract_json(text: str) -> str | None:
        """Extract the first JSON object from text that may contain preamble."""
        # Try the whole string first (grammar-guided output)
        try:
            json.loads(text)
            return text
        except (json.JSONDecodeError, ValueError):
            pass
        # Find the first { ... } block (unguided output with chain-of-thought)
        start = text.find("{")
        if start == -1:
            return None
        # Find matching closing brace
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return None

