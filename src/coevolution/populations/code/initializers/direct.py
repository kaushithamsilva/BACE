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
from ..helpers.llm_helpers import CodeLLMHelpers


@initializer_registry.register("direct", population="code")
class DirectCodeInitializer(CodeLLMHelpers, BaseLLMInitializer[CodeIndividual]):
    """Creates Gen-0 code individuals via zero-shot LLM generation."""

    def initializer_name(self) -> str:
        return "direct"

    def initialize(self, problem: Problem, size: int | None = None) -> list[CodeIndividual]:
        # This initializer produces individuals one-at-a-time. Parallelism 
        # is managed by the WeightedPopulationInitializer orchestrator.
        prompt = self.prompt_manager.render_prompt(
            "initialization/direct/initial_single.j2",
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
                metadata={"initializer": self.initializer_name()},
            )
        ]
