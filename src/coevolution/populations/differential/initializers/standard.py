"""BootstrappedInitializer — differential population always starts empty."""

from __future__ import annotations

from loguru import logger

from coevolution.core.interfaces import Problem
from coevolution.core.individual import TestIndividual

from coevolution.strategies.llm_base import BaseLLMInitializer
from coevolution.populations.registries import initializer_registry


@initializer_registry.register("bootstrapped", population="differential")
class BootstrappedInitializer(BaseLLMInitializer[TestIndividual]):
    """Differential tests start empty — Gen 0 is always []."""

    def initializer_name(self) -> str:
        return "bootstrapped"

    def initialize(self, problem: Problem, size: int | None = None) -> list[TestIndividual]:
        logger.debug("BootstrappedInitializer: starting with empty population")
        return []


__all__ = ["BootstrappedInitializer"]
