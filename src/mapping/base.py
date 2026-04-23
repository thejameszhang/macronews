from abc import ABC, abstractmethod
from mapping.schemas import SingleAssetResult


class BaseMapper(ABC):
    """Abstract base class for (article, asset) relevance mappers.

    A mapper decides whether a single asset is affected by a single article
    and returns the evidence, signal, and score. Kept abstract so alternate
    backends (e.g., knowledge-graph mappers) can plug in alongside the LLM
    implementation.
    """

    @abstractmethod
    def map_single_asset(
        self,
        texts: list[str],
        max_tokens: int = 512,
    ) -> list[SingleAssetResult]:
        """Classify a batch of (article, asset) prompts, one result per prompt."""
        ...
