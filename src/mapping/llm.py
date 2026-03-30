import json
import logging

import compat  # noqa: F401 — transformers 5.x / vLLM shim

from config.paths import PROMPTS_DIR
from mapping.base import BaseMapper
from mapping.schemas import (
    ASSET_SYMBOLS_SET,
    AssetMapping,
    NAME_TO_SYMBOL,
    MappingResult,
    ValidationResult,
)

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (PROMPTS_DIR / "headline.txt").read_text()


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

    @property
    def system_prompt(self) -> str:
        """Inject the dynamic asset list into the system prompt template."""
        return self.system_prompt_template.replace("{{ASSET_LIST}}", self._asset_list_str)

    @system_prompt.setter
    def system_prompt(self, value: str):
        self.system_prompt_template = value

    def _get_formatted_asset_list(self) -> str:
        """Build a hierarchical string representation of the full asset universe."""
        from utils.config import load_asset_universe
        from config.paths import ASSET_UNIVERSE_YAML
        
        universe = load_asset_universe(ASSET_UNIVERSE_YAML)
        
        # Group by asset_class
        by_class = {}
        for sym, info in universe.items():
            cls = info.get("asset_class", "other")
            by_class.setdefault(cls, []).append((sym, info.get("name", sym)))
            
        lines = []
        
        # 1. GICS Sectors
        if "equity_sector" in by_class:
            lines.append("**EQUITY SECTORS (GICS)**")
            sectors = [f"{sym} ({name})" for sym, name in sorted(by_class["equity_sector"])]
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
        if "bond" in by_class:
            lines.append("**BONDS**")
            # Sub-group bonds by region
            us = []
            eu = []
            other = []
            for sym, name in by_class["bond"]:
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
        if "equity" in by_class:
            lines.append("**EQUITY INDEX FUTURES**")
            us = []
            eu = []
            other = []
            for sym, name in by_class["equity"]:
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
        if "stir" in by_class:
            lines.append("**SHORT-TERM INTEREST RATE FUTURES (STIR)**")
            stirs = [f"{sym} ({name})" for sym, name in sorted(by_class["stir"])]
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
            )

    def _map_vllm(self, texts: list[str], schema_class=None, max_tokens: int = 256,
                  guided: bool = True) -> list[MappingResult]:
        from vllm import SamplingParams
        self._init_llm()

        if guided:
            from vllm.sampling_params import StructuredOutputsParams
            cls = schema_class or MappingResult
            schema = cls.model_json_schema()
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=max_tokens,
                structured_outputs=StructuredOutputsParams(json=schema),
            )
        else:
            # Unguided: no grammar constraint, rely on prompt + temperature=0.
            # repetition_penalty=1.1 prevents degeneration loops on long inputs.
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=max_tokens,
                repetition_penalty=1.1,
            )

        conversations = [self._build_conversation(t) for t in texts]
        logger.info(f"[vLLM] Generating {len(conversations):,} mappings...")
        outputs = self._llm.chat(conversations, sampling_params)

        results = []
        total_dropped = 0
        articles_with_drops = 0

        for i, output in enumerate(outputs):
            raw = output.outputs[0].text
            logger.info("[vLLM] Output %d (len=%d): %s", i, len(raw), raw[:1000])
            result, n_dropped = self._parse_lenient(raw)
            results.append(result)
            if n_dropped > 0:
                total_dropped += n_dropped
                articles_with_drops += 1

        if total_dropped > 0:
            logger.warning(f"Dropped {total_dropped} invalid asset symbols across {articles_with_drops} articles.")

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

    def _parse_lenient(self, raw: str | None) -> tuple[MappingResult, int]:
        if raw is None:
            return MappingResult(), 0
        json_str = self._extract_json(raw)
        if json_str is None:
            logger.debug("No JSON found in output: %s", raw[:200])
            return MappingResult(), 0
        try:
            data = json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            logger.debug("JSON parse failed: %s", json_str[:200])
            return MappingResult(), 0

        if not isinstance(data, dict):
            return MappingResult(), 0

        # Parse relevant_assets (supports both list[str] and list[dict])
        raw_assets = data.get("relevant_assets", [])
        if not isinstance(raw_assets, list):
            return MappingResult(), 0

        valid: list[AssetMapping] = []
        seen: set[str] = set()
        n_dropped = 0
        for item in raw_assets:
            # Extract symbol, signal, and reasoning from either format
            if isinstance(item, dict):
                sym_raw = item.get("asset", "")
                signal = item.get("signal", "strong")
                if signal not in ("strong", "weak"):
                    signal = "strong"
                reasoning = item.get("reasoning", "") or item.get("rationale", "")
            elif isinstance(item, str):
                sym_raw = item
                signal = "strong"
                reasoning = ""
            else:
                n_dropped += 1
                continue

            sym_stripped = sym_raw.strip() if isinstance(sym_raw, str) else ""
            if sym_stripped in ASSET_SYMBOLS_SET:
                sym = sym_stripped
            elif sym_stripped.lower() in NAME_TO_SYMBOL:
                sym = NAME_TO_SYMBOL[sym_stripped.lower()]
            else:
                n_dropped += 1
                continue

            if sym not in seen:
                valid.append(AssetMapping(asset=sym, signal=signal, reasoning=reasoning))
                seen.add(sym)

        # Parse company_specific (optional, only from article-level)
        company_specific = data.get("company_specific", False)
        if not isinstance(company_specific, bool):
            company_specific = False

        # Parse macro_summary (optional, only from article-level v2)
        macro_summary = data.get("macro_summary", "")
        if not isinstance(macro_summary, str):
            macro_summary = ""

        return MappingResult(relevant_assets=valid,
                             company_specific=company_specific, macro_summary=macro_summary), n_dropped

    def map(self, headlines: list[str], schema_class=None, max_tokens: int = 256,
            guided: bool = True) -> list[MappingResult]:
        return self._map_vllm(headlines, schema_class=schema_class,
                              max_tokens=max_tokens, guided=guided)

    def map_validate(
        self,
        texts: list[str],
        max_tokens: int = 256,
    ) -> list[ValidationResult]:
        """Stage 3 validation: returns ValidationResult objects via guided decoding."""
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        self._init_llm()

        result_class = ValidationResult
        schema = result_class.model_json_schema()
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            structured_outputs=StructuredOutputsParams(json=schema),
        )

        conversations = [self._build_conversation(t) for t in texts]
        logger.info(f"[vLLM] Validating {len(conversations):,} (article, asset) pairs...")
        outputs = self._llm.chat(conversations, sampling_params)

        results = []
        for i, output in enumerate(outputs):
            raw = output.outputs[0].text
            logger.info("[vLLM] Validation %d (len=%d): %s", i, len(raw), raw[:500])
            try:
                result = result_class.model_validate_json(raw)
            except Exception:
                json_str = self._extract_json(raw)
                if json_str:
                    try:
                        result = result_class.model_validate_json(json_str)
                    except Exception:
                        logger.warning("Failed to parse validation output %d: %s", i, raw[:200])
                        result = result_class(valid=False)
                else:
                    logger.warning("No JSON in validation output %d: %s", i, raw[:200])
                    result = result_class(valid=False)
            results.append(result)

        return results
