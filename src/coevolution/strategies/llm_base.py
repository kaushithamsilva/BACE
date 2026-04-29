"""Shared base classes and helpers for LLM-backed evolutionary operators.

This module contains protocol definitions, retry decorator factory,
and the `BaseLLMService` class which handles LLM orchestration.

Prompt Management:
    Templates are loaded from multiple directories in the following order:

    1. The 'prompts/' folder where the operator's code is defined.
       (e.g. 'unittest/prompts/' for unittest-based operators).

    2. The 'prompts/' folder of the population being modified.
       (e.g. 'code/prompts/' when repairing code).
       This is explicitly defined for specialized repair operators which some test populations may bring in.

    3. The global 'src/coevolution/prompts/' folder for shared snippets.
"""

import inspect
import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Protocol, Tuple, Type

from loguru import logger
from tenacity import (
    WrappedFn,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from coevolution.core.interfaces import (
    BaseIndividual,
    CoevolutionContext,
    IPopulationInitializer,
    Problem,
)
from coevolution.core.interfaces.language import ICodeParser
from coevolution.core.interfaces.operators import IOperator
from coevolution.core.interfaces.probability import IProbabilityAssigner
from coevolution.core.interfaces.selection import IParentSelectionStrategy
from coevolution.utils.prompt_manager import PromptManager


def llm_retry(
    exception_types: Tuple[Type[Exception], ...],
) -> Callable[[WrappedFn], WrappedFn]:
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(exception_types),
        reraise=True,
    )


class ILanguageModel(Protocol):
    """Protocol defining the minimal interface for language models."""

    def generate(self, prompt: str) -> str:
        """Generate text response from the language model."""
        ...


class LLMGenerationError(Exception):
    """Raised when LLM fails to generate output."""


class LLMSyntaxError(Exception):
    """Raised when LLM generates code with invalid syntax."""


class BaseLLMService:
    """Abstract base service providing LLM orchestration.

    Handles prompt rendering tools, LLM calls, retries, and code block extraction.
    """

    def __init__(
        self,
        llm: ILanguageModel,
        parser: ICodeParser,
        language_name: str,
        population_name: Optional[str] = None,
    ) -> None:
        self._llm = llm
        self.parser = parser
        self.language_name = language_name
        self.population_name = population_name

        # Resolve prioritized template search paths
        template_dirs = self._resolve_template_dirs(population_name)
        self.prompt_manager = PromptManager(
            template_dirs=template_dirs, language=language_name
        )

        logger.debug(
            f"Initialized {self.__class__.__name__} for pop={population_name} "
            f"with search paths: {[os.path.basename(os.path.dirname(p)) for p in template_dirs]}"
        )

    def _resolve_template_dirs(self, target_pop: Optional[str]) -> list[str]:
        """Resolves the prioritized search paths for Jinja2 templates."""
        # src/coevolution/strategies/llm_base.py -> src/coevolution/
        base_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dirs = []

        # 1. Home population prompts (where the class is physically defined)
        # Highest priority for specialized operators
        try:
            class_file = inspect.getfile(self.__class__)
            # populations/<pop_name>/operators/file.py -> populations/<pop_name>/
            pop_root = os.path.dirname(os.path.dirname(class_file))
            home_prompts = os.path.join(pop_root, "prompts")
            if os.path.isdir(home_prompts):
                dirs.append(home_prompts)
        except (TypeError, ValueError):
            pass

        # 2. Target population prompts (the population the operator is acting on)
        if target_pop:
            target_path = os.path.join(base_src, "populations", target_pop, "prompts")
            if os.path.isdir(target_path):
                dirs.append(target_path)

        # 3. Global Prompts
        global_prompts = os.path.join(base_src, "prompts")
        dirs.append(global_prompts)

        # Deduplicate while preserving priority order
        seen = set()
        unique_dirs = []
        for d in dirs:
            # Normalize path for comparison
            norm_d = os.path.normpath(d)
            if norm_d not in seen:
                unique_dirs.append(d)
                seen.add(norm_d)

        return unique_dirs

    @llm_retry(exception_types=(LLMGenerationError,))
    def _generate(self, prompt: Any) -> str:
        logger.debug("Sending prompt to LLM")
        logger.trace(f"Prompt preview: {prompt[:100]}...")

        try:
            raw_response = self._llm.generate(prompt)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            logger.debug(f"Prompt that caused failure: {prompt}")
            raise LLMGenerationError(f"LLM API call failed: {e}") from e

        if not raw_response or not raw_response.strip():
            logger.warning("LLM returned empty response")
            raise LLMGenerationError("LLM returned empty response")

        logger.debug("Received response from LLM")
        logger.trace(f"Raw response preview: {raw_response[:100]}...")

        return raw_response

    def _extract_code_block(self, response: str) -> str:
        """
        Extract code block from LLM response using language adapter.

        Args:
            response: Raw text response from the LLM

        Returns:
            Extracted code block or the original response if no block found
        """
        blocks = self.parser.extract_code_blocks(response)
        if blocks:
            return blocks[0]
        return response

    def _validate_syntax(self, code: str) -> None:
        """
        Validate syntax of the provided code using the language parser.

        Args:
            code: The code to validate

        Raises:
            LLMSyntaxError: If the syntax is invalid
        """
        if not self.parser.is_syntax_valid(code):
            raise LLMSyntaxError("Generated code has invalid syntax")


class BaseLLMOperator[T: BaseIndividual](BaseLLMService, IOperator[T], ABC):
    """
    Base class for all LLM-backed genetic operators (Mutation, Crossover, Edit).

    Extends IOperator with LLM generation capabilities and standard evolutionary
    dependencies: parent selection and probability assignment.
    Concrete classes implement `execute()`.
    """

    def __init__(
        self,
        llm: ILanguageModel,
        parser: ICodeParser,
        language_name: str,
        parent_selector: IParentSelectionStrategy[T],
        prob_assigner: IProbabilityAssigner,
        population_name: Optional[str] = None,
    ) -> None:
        super().__init__(llm, parser, language_name, population_name=population_name)
        self.parent_selector = parent_selector
        self.prob_assigner = prob_assigner

    @abstractmethod
    def operation_name(self) -> str: ...

    @abstractmethod
    def execute(self, context: CoevolutionContext) -> list[T]: ...

    def reset(self, scope: str = "problem") -> None:
        """Default no-op reset for LLM operators."""
        pass


class BaseLLMInitializer[T: BaseIndividual](
    BaseLLMService, IPopulationInitializer[T], ABC
):
    """
    Base class for all population initializers (Gen 0 creators).

    Combines LLM generation capabilities with a population configuration.
    Concrete classes implement `initialize()`.
    """

    def __init__(
        self,
        llm: ILanguageModel,
        parser: ICodeParser,
        language_name: str,
        population_name: Optional[str] = None,
    ) -> None:
        super().__init__(llm, parser, language_name, population_name=population_name)

    @abstractmethod
    def initialize(self, problem: Problem, size: int | None = None) -> list[T]: ...


__all__ = [
    "ILanguageModel",
    "llm_retry",
    "LLMGenerationError",
    "LLMSyntaxError",
    "BaseLLMService",
    "BaseLLMOperator",
    "BaseLLMInitializer",
]
