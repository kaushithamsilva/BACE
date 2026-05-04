"""PublicCodeRepairOperator — specialized code repair using public/anchor test feedback."""

from __future__ import annotations

from typing import Any

from loguru import logger
from coevolution.core.individual import CodeIndividual
from coevolution.core.interfaces import (
    CoevolutionContext,
    LanguageParsingError,
    LanguageTransformationError,
)
from coevolution.core.interfaces.language import ICodeParser
from coevolution.strategies.llm_base import (
    ILanguageModel,
    LLMGenerationError,
    LLMSyntaxError,
    llm_retry,
)
from coevolution.core.interfaces.probability import IProbabilityAssigner
from coevolution.core.interfaces.selection import IParentSelectionStrategy
from coevolution.populations.code.operators.repair import CodeGenericRepairOperator
from coevolution.populations.registries import operator_registry
from coevolution.strategies.selection.failing_test_selection import FailingTestSelector


@operator_registry.register("public_repair", population="code")
class PublicCodeRepairOperator(CodeGenericRepairOperator):
    """Specialized 3-step repair process for public tests: Simulation -> Plan -> Code."""

    def __init__(
        self,
        llm: ILanguageModel,
        parser: ICodeParser,
        language_name: str,
        parent_selector: IParentSelectionStrategy[CodeIndividual],
        prob_assigner: IProbabilityAssigner,
        failing_test_selector: type[FailingTestSelector] = FailingTestSelector,
        k_failing_tests: int = 3,  # Capped at 3 as per requirements
        **kwargs: Any,
    ) -> None:
        super().__init__(
            llm=llm,
            parser=parser,
            language_name=language_name,
            parent_selector=parent_selector,
            prob_assigner=prob_assigner,
            failing_test_selector=failing_test_selector,
            k_failing_tests=k_failing_tests,
            target_test_type="public",
        )

    def operation_name(self) -> str:
        return "public_repair"

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
            logger.warning("PublicCodeRepairOperator: no parents available")
            return []
        parent = parents[0]

        # Select k failing tests (capped at 3)
        failing = self._failing_test_selector.select_k_failing_tests(
            context, parent, k=self.k_failing_tests, test_type_filter=self.target_test_type
        )
        if not failing:
            logger.debug(
                "PublicCodeRepairOperator: no failing tests for this parent, skipping"
            )
            return []

        failing_tests_data = []
        for test_ind, test_pop_type in failing:
            exec_result = context.interactions[test_pop_type].execution_results
            trace = "No trace available"
            if parent.id in exec_result and test_ind.id in exec_result[parent.id]:
                trace = exec_result[parent.id][test_ind.id].error_log or trace
            failing_tests_data.append({"snippet": test_ind.snippet, "trace": trace})

        # Step 1: Simulate
        simulation = self._call_simulate(problem, parent, failing_tests_data)

        # Step 2: Plan
        plan = self._call_plan(problem, parent, failing_tests_data, simulation)

        # Step 3: Code
        edited_code = self._call_generate(problem, parent, simulation, plan)

        probability = self.prob_assigner.assign_probability(
            self.operation_name(), [parent.probability]
        )
        return [
            CodeIndividual(
                snippet=edited_code,
                probability=probability,
                creation_op=self.operation_name(),
                generation_born=code_pop.generation + 1,
                parents={
                    "code": [parent.id],
                    "test": [t.id for t, _ in failing],
                },
                explanation=plan, # Store the plan as explanation
                metadata={
                    "num_failing_tests": len(failing),
                    "target_test_type": self.target_test_type,
                    "simulation": simulation,
                },
            )
        ]

    def _call_simulate(self, problem: Any, parent: CodeIndividual, failing_tests: list[dict[str, str]]) -> str:
        prompt = self.prompt_manager.render_prompt(
            "repair/simulation.j2",
            question_content=problem.question_content,
            individual=parent.snippet,
            failing_tests=failing_tests,
        )
        return self._generate(prompt)

    def _call_plan(self, problem: Any, parent: CodeIndividual, failing_tests: list[dict[str, str]], simulation: str) -> str:
        prompt = self.prompt_manager.render_prompt(
            "repair/plan.j2",
            question_content=problem.question_content,
            individual=parent.snippet,
            failing_tests=failing_tests,
            simulation=simulation,
        )
        return self._generate(prompt)

    def _call_generate(self, problem: Any, parent: CodeIndividual, simulation: str, plan: str) -> str:
        prompt = self.prompt_manager.render_prompt(
            "repair/code.j2",
            question_content=problem.question_content,
            starter_code=problem.starter_code,
            individual=parent.snippet,
            simulation=simulation,
            plan=plan,
        )
        response = self._generate(prompt)
        edited_code = self._extract_code_block(response)
        return self._validated_code(edited_code, problem.starter_code, "edit")
