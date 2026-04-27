"""Planning Code Initializer."""

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


@initializer_registry.register("planning", population="code")
class PlanningCodeInitializer(_CodeLLMHelpers, BaseLLMInitializer[CodeIndividual]):
    """Creates Gen-0 code individuals via a two-step Plan-then-Code process."""

    def __init__(
        self,
        llm: ILanguageModel,
        parser: ICodeParser,
        language_name: str,
    ) -> None:
        super().__init__(llm, parser, language_name)

    def initialize(self, problem: Problem, size: int | None = None) -> list[CodeIndividual]:
        # Step 1: Generate Plan
        plan_prompt = self.prompt_manager.render_prompt(
            "operators/code/plan_generate.j2",
            question_content=problem.question_content,
            starter_code=problem.starter_code,
        )
        plan = self._generate(plan_prompt)
        
        # Step 2: Generate Code from Plan
        code_prompt = self.prompt_manager.render_prompt(
            "operators/code/plan_to_code.j2",
            question_content=problem.question_content,
            starter_code=problem.starter_code,
            plan=plan
        )
        response = self._generate(code_prompt)
        snip = self._extract_code_block(response)
        snip = self._validated_code(snip, problem.starter_code, "initial")
        
        return [
            CodeIndividual(
                snippet=snip,
                probability=0.0,  # Set by orchestrator
                creation_op=OPERATION_INITIAL,
                generation_born=0,
                explanation=self.parser.get_docstring(snip),
                metadata={
                    "initializer": self.__class__.__name__,
                    "plan": plan
                },
            )
        ]
