from abc import ABC, abstractmethod
from mapping.schemas import MappingResult

class BaseMapper(ABC):
    """
    Abstract base class for mapping news headlines to macro assets.
    """
    
    @abstractmethod
    def map(self, headlines: list[str]) -> list[MappingResult]:
        """
        Maps a list of headlines to relevant macro asset symbols.
        """
        pass
