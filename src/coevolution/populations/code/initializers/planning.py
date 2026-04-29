"""Planning Code Initializer."""

from __future__ import annotations

from coevolution.core.individual import CodeIndividual
from coevolution.core.interfaces import (
    OPERATION_INITIAL,
    Problem,
)
from coevolution.core.interfaces.language import ICodeParser
from coevolution.strategies.llm_base import (
    BaseLLMInitializer,
    ILanguageModel,
    LLMGenerationError,
    LLMSyntaxError,
    llm_retry,
)
from coevolution.populations.registries import initializer_registry
from ..helpers.llm_helpers import CodeLLMHelpers


@initializer_registry.register("planning", population="code")
class PlanningInitializer(CodeLLMHelpers, BaseLLMInitializer[CodeIndividual]):
    """Creates Gen-0 code individuals via a two-step Plan-then-Code process."""

    def initializer_name(self) -> str:
        return "planning"

    def initialize(self, problem: Problem, size: int | None = None) -> list[CodeIndividual]:
        # Step 1: Generate Plan
        plan = self._call_generate_plan(problem)

        # Step 2: Generate Code from Plan
        snippet = self._call_generate_code(problem, plan)

        return [
            CodeIndividual(
                snippet=snippet,
                probability=0.0,  # Set by orchestrator
                creation_op=OPERATION_INITIAL,
                generation_born=0,
                explanation=plan,  # Store the plan in the explanation field
                metadata={
                    "initializer": self.initializer_name(),
                },
            )
        ]

    @llm_retry((LLMGenerationError,))
    def _call_generate_plan(self, problem: Problem) -> str:
        """Call LLM to generate an algorithmic plan."""
        plan_prompt = self.prompt_manager.render_prompt(
            "initialization/planning/plan_generate.j2",
            question_content=problem.question_content,
            starter_code=problem.starter_code,
        )
        return self._generate(plan_prompt)

    @llm_retry((LLMGenerationError, LLMSyntaxError, ValueError))
    def _call_generate_code(self, problem: Problem, plan: str) -> str:
        """Call LLM to generate code based on a plan."""
        code_prompt = self.prompt_manager.render_prompt(
            "initialization/planning/plan_to_code.j2",
            question_content=problem.question_content,
            starter_code=problem.starter_code,
            plan=plan,
        )
        response = self._generate(code_prompt)
        snip = self._extract_code_block(response)
        # _validated_code handles syntax validation and starter code check
        return self._validated_code(snip, problem.starter_code, "initial_planning")
