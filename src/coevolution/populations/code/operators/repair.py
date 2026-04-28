"""CodeGenericRepairOperator — guides code repair using failing tests."""

from __future__ import annotations



from loguru import logger
from coevolution.populations.registries import operator_registry

from coevolution.core.individual import CodeIndividual
from coevolution.core.interfaces import (
    CoevolutionContext,
    LanguageParsingError,
    LanguageTransformationError,
)
from coevolution.core.interfaces.language import ICodeParser
from coevolution.core.interfaces.probability import IProbabilityAssigner
from coevolution.core.interfaces.selection import IParentSelectionStrategy
from coevolution.core.interfaces.types import OPERATION_GENERIC_REPAIR
from coevolution.strategies.llm_base import (
    BaseLLMOperator,
    ILanguageModel,
    LLMGenerationError,
    LLMSyntaxError,
    llm_retry,
)
from coevolution.strategies.selection.failing_test_selection import FailingTestSelector

from ._helpers import _CodeLLMHelpers

type TestPopulationType = str


@operator_registry.register("generic_repair", population="code")
class CodeGenericRepairOperator(_CodeLLMHelpers, BaseLLMOperator[CodeIndividual]):
    """Self-sufficient operator for feedback-driven mutation (Repair).

    On each execute(context) call:
      1. Selects K failing tests (across all test populations).
      2. Selects 1 parent CodeIndividual using RouletteWheelSelection.
      3. Calls LLM to 'fix' the code based on the failure feedback.
      4. Assigns probability to offspring.
      5. Returns exactly one new CodeIndividual.
    """

    def __init__(
        self,
        llm: ILanguageModel,
        parser: ICodeParser,
        language_name: str,
        parent_selector: IParentSelectionStrategy[CodeIndividual],
        prob_assigner: IProbabilityAssigner,
        failing_test_selector: type[FailingTestSelector] = FailingTestSelector,
        k_failing_tests: int = 10,
        target_test_type: str | None = None,
    ) -> None:
        super().__init__(llm, parser, language_name, parent_selector, prob_assigner)
        self.k_failing_tests = k_failing_tests
        self._failing_test_selector = failing_test_selector
        self.target_test_type = target_test_type

    def operation_name(self) -> str:
        return OPERATION_GENERIC_REPAIR

    @llm_retry(
        (
            ValueError,
            LanguageParsingError,
            LanguageTransformationError,
            LLMGenerationError,
            LLMSyntaxError,
        )
    )
    def execute(self, context: CoevolutionContext) -> list[CodeIndividual]:
        code_pop = context.code_population
        problem = context.problem

        parents = self.parent_selector.select_parents(code_pop, 1, context)
        if not parents:
            logger.warning("CodeGenericRepairOperator: no parents available")
            return []
        parent = parents[0]

        failing = self._failing_test_selector.select_k_failing_tests(
            context, parent, k=self.k_failing_tests, test_type_filter=self.target_test_type
        )
        if not failing:
            logger.debug(
                "CodeGenericRepairOperator: no failing tests for this parent, skipping"
            )
            return []
        failing_tests_data = []
        for test_ind, test_pop_type in failing:
            exec_result = context.interactions[test_pop_type].execution_results
            trace = "No trace available"
            if parent.id in exec_result and test_ind.id in exec_result[parent.id]:
                trace = exec_result[parent.id][test_ind.id].error_log or trace
            failing_tests_data.append({"snippet": test_ind.snippet, "trace": trace})

        prompt = self.prompt_manager.render_prompt(
            "operators/code/edit.j2",
            question_content=problem.question_content,
            starter_code=problem.starter_code,
            individual=parent.snippet,
            failing_tests=failing_tests_data,
        )
        response = self._generate(prompt)
        edited_code = self._extract_code_block(response)
        edited_code = self._validated_code(edited_code, problem.starter_code, "edit")

        probability = self.prob_assigner.assign_probability(
            OPERATION_GENERIC_REPAIR, [parent.probability]
        )
        return [
            CodeIndividual(
                snippet=edited_code,
                probability=probability,
                creation_op=OPERATION_GENERIC_REPAIR,
                generation_born=code_pop.generation + 1,
                parents={
                    "code": [parent.id],
                    "test": [t.id for t, _ in failing],
                },
                explanation=self.parser.get_docstring(edited_code),
                metadata={
                    "num_failing_tests": len(failing),
                    "target_test_type": self.target_test_type,
                },
            )
        ]


__all__ = ["CodeGenericRepairOperator"]
