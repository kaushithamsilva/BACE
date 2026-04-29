"""Direct Code Initializer."""

from __future__ import annotations

from coevolution.core.individual import CodeIndividual
from coevolution.core.interfaces import (
    OPERATION_INITIAL,
    Problem,
)
from coevolution.core.interfaces.language import ICodeParser
from coevolution.strategies.llm_base import BaseLLMInitializer, ILanguageModel
from coevolution.populations.registries import initializer_registry
from ..operators._helpers import _CodeLLMHelpers


@initializer_registry.register("direct", population="code")
class DirectCodeInitializer(_CodeLLMHelpers, BaseLLMInitializer[CodeIndividual]):
    """Creates Gen-0 code individuals via zero-shot LLM generation."""

    def __init__(
        self,
        llm: ILanguageModel,
        parser: ICodeParser,
        language_name: str,
    ) -> None:
        super().__init__(llm, parser, language_name)

    def initialize(self, problem: Problem, size: int | None = None) -> list[CodeIndividual]:
        # This initializer produces individuals one-at-a-time. Parallelism 
        # is managed by the WeightedPopulationInitializer orchestrator.
        prompt = self.prompt_manager.render_prompt(
            "initialization/initial_single.j2",
            question_content=problem.question_content,
            starter_code=problem.starter_code,
        )
        response = self._generate(prompt)
        snip = self._extract_code_block(response)
        snip = self._validated_code(snip, problem.starter_code, "initial")
        
        return [
            CodeIndividual(
                snippet=snip,
                probability=0.0,  # Set by orchestrator
                creation_op=OPERATION_INITIAL,
                generation_born=0,
                explanation=self.parser.get_docstring(snip),
                metadata={"initializer": self.__class__.__name__},
            )
        ]
