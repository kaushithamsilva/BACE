"""Base functionality for Code initializers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from coevolution.core.individual import CodeIndividual
from coevolution.core.interfaces import (
    PopulationConfig,
    Problem,
)
from coevolution.core.interfaces.language import ICodeParser
from coevolution.strategies.llm_base import (
    BaseLLMInitializer,
    ILanguageModel,
)

from ..operators._helpers import _CodeLLMHelpers


class BaseCodeInitializer(_CodeLLMHelpers, BaseLLMInitializer[CodeIndividual], ABC):
    """Base class for code population initializers."""

    def __init__(
        self,
        llm: ILanguageModel,
        parser: ICodeParser,
        language_name: str,
        pop_config: PopulationConfig,
        llm_workers: int = 4,
    ) -> None:
        super().__init__(llm, parser, language_name, pop_config)
        self.llm_workers = llm_workers

    @abstractmethod
    def initialize(self, problem: Problem, size: int | None = None) -> list[CodeIndividual]:
        """Create Gen-0 code individuals."""
        ...
