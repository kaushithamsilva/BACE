"""UnittestInitializer — creates Gen-0 test individuals via LLM."""

from __future__ import annotations

from coevolution.core.individual import TestIndividual
from coevolution.core.interfaces import (
    OPERATION_INITIAL,
    Problem,
)
from coevolution.populations.registries import initializer_registry
from coevolution.core.interfaces.language import (
    ICodeParser,
    LanguageParsingError,
    LanguageTransformationError,
)

from coevolution.strategies.llm_base import (
    BaseLLMInitializer,
    ILanguageModel,
    LLMGenerationError,
    LLMSyntaxError,
    llm_retry,
)
from ..operators._helpers import _TestLLMHelpers


@initializer_registry.register("unittest", population="unittest")
class UnittestInitializer(_TestLLMHelpers, BaseLLMInitializer[TestIndividual]):
    """Creates Gen-0 test individuals via LLM.

    Asks for `population_size` tests in one shot and recovers
    gracefully if the count doesn't match.
    """

    def __init__(
        self,
        llm: ILanguageModel,
        parser: ICodeParser,
        language_name: str,
        yields_per_call: int = 20,
    ) -> None:
        super().__init__(llm, parser, language_name)
        self._yields_per_call = yields_per_call

    @property
    def yields_per_call(self) -> int:
        return self._yields_per_call

    def initialize(self, problem: Problem, size: int | None = None) -> list[TestIndividual]:
        if size is None:
            size = self._yields_per_call
        test_functions = self._generate_test_functions(problem, size)

        individuals: list[TestIndividual] = []
        for fn in test_functions:
            individuals.append(
                TestIndividual(
                    snippet=fn,
                    probability=0.0,  # Set by orchestrator
                    creation_op=OPERATION_INITIAL,
                    generation_born=0,
                    explanation=self.parser.get_docstring(fn),
                    metadata={"initializer": self.__class__.__name__},
                )
            )
        return individuals

    @llm_retry(
        (
            ValueError,
            LanguageParsingError,
            LanguageTransformationError,
            LLMGenerationError,
            LLMSyntaxError,
        )
    )
    def _generate_test_functions(self, problem: Problem, target: int) -> list[str]:
        prompt = self.prompt_manager.render_prompt(
            "operators/unittest/initial.j2",
            population_size=target,
            question_content=problem.question_content,
            starter_code=problem.starter_code,
        )
        response = self._generate(prompt)

        code_blocks = self.parser.extract_code_blocks(response)
        test_functions: list[str] = []
        for block in code_blocks:
            clean_block = self.parser.remove_main_block(block)
            test_functions.extend(self._extract_test_functions(clean_block))

        # Trim over-generation
        if len(test_functions) > target:
            test_functions = test_functions[:target]

        # Top-up under-generation with an additional LLM call if necessary
        if len(test_functions) < target:
            additional = target - len(test_functions)
            extra_prompt = self.prompt_manager.render_prompt(
                "operators/unittest/initial.j2",
                population_size=additional,
                question_content=problem.question_content,
                starter_code=problem.starter_code,
            )
            extra_response = self._generate(extra_prompt)
            for block in self.parser.extract_code_blocks(extra_response):
                clean_block = self.parser.remove_main_block(block)
                test_functions.extend(self._extract_test_functions(clean_block))
                if len(test_functions) >= target:
                    break
            test_functions = test_functions[:target]

        return test_functions


__all__ = ["UnittestInitializer"]
