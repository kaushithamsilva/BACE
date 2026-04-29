"""DryRUN Code Initializer."""

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


@initializer_registry.register("DryRUN", population="code")
class DryRUNInitializer(CodeLLMHelpers, BaseLLMInitializer[CodeIndividual]):
    """Creates Gen-0 code individuals via a multi-step Plan-then-Code-then-Simulate process.

    Workflow:
    1. Problem Modification (Remove examples)
    2. Initial Plan Generation
    3. Plan Refinement (iterations)
    4. Initial Code Generation
    5. Simulation & Plan/Code Refinement Loop (iterations)
    6. Final Code Refinement (iterations)
    """

    def __init__(
        self,
        llm: ILanguageModel,
        parser: ICodeParser,
        language_name: str,
        plan_iterations: int = 2,
        sim_iterations: int = 2,
        final_refine_iterations: int = 1,
    ) -> None:
        super().__init__(llm, parser, language_name)
        self.plan_iterations = plan_iterations
        self.sim_iterations = sim_iterations
        self.final_refine_iterations = final_refine_iterations

    def initializer_name(self) -> str:
        return "DryRUN"

    def initialize(self, problem: Problem, size: int | None = None) -> list[CodeIndividual]:
        # Step 1: Modify Problem
        modified_problem = self._call_modify_problem(problem)

        # Step 2: Initial Plan
        current_plan = self._call_generate_plan(problem, modified_problem)

        # Step 3: Plan Refinement
        for _ in range(self.plan_iterations):
            current_plan = self._call_refine_plan(problem, modified_problem, current_plan)

        # Step 4: Initial Code Generation
        current_code = self._call_generate_code(problem, modified_problem, current_plan)

        # Step 5: Simulate and Refine Plan Loop
        simulation_traces = []
        for _ in range(self.sim_iterations):
            trace = self._call_simulate(problem, modified_problem, current_code)
            simulation_traces.append(trace)
            
            current_plan = self._call_refine_plan_with_sim(
                problem, modified_problem, current_plan, trace
            )
            current_code = self._call_generate_code(problem, modified_problem, current_plan)

        # Step 6: Final Code Refinement
        for _ in range(self.final_refine_iterations):
            current_code = self._call_refine_code(
                problem, modified_problem, current_plan, current_code
            )

        return [
            CodeIndividual(
                snippet=current_code,
                probability=0.0,  # Set by orchestrator
                creation_op=OPERATION_INITIAL,
                generation_born=0,
                explanation=current_plan,
                metadata={
                    "initializer": self.initializer_name(),
                    "modified_problem": modified_problem,
                    "simulation_traces": simulation_traces,
                },
            )
        ]

    @llm_retry((LLMGenerationError,))
    def _call_modify_problem(self, problem: Problem) -> str:
        prompt = self.prompt_manager.render_prompt(
            "initialization/dryrun/modify_problem.j2",
            question_content=problem.question_content,
        )
        return self._generate(prompt)

    @llm_retry((LLMGenerationError,))
    def _call_generate_plan(self, problem: Problem, modified_problem: str) -> str:
        prompt = self.prompt_manager.render_prompt(
            "initialization/dryrun/plan_generate.j2",
            question_content=modified_problem,
            starter_code=problem.starter_code,
        )
        return self._generate(prompt)

    @llm_retry((LLMGenerationError,))
    def _call_refine_plan(
        self, problem: Problem, modified_problem: str, previous_plan: str
    ) -> str:
        prompt = self.prompt_manager.render_prompt(
            "initialization/dryrun/plan_refine.j2",
            question_content=modified_problem,
            starter_code=problem.starter_code,
            previous_plan=previous_plan,
        )
        return self._generate(prompt)

    @llm_retry((LLMGenerationError, LLMSyntaxError, ValueError))
    def _call_generate_code(
        self, problem: Problem, modified_problem: str, plan: str
    ) -> str:
        prompt = self.prompt_manager.render_prompt(
            "initialization/dryrun/solution_generate.j2",
            question_content=modified_problem,
            plan=plan,
            starter_code=problem.starter_code,
            language=self.language_name,
        )
        response = self._generate(prompt)
        snip = self._extract_code_block(response)
        return self._validated_code(snip, problem.starter_code, "dryrun_initial")

    @llm_retry((LLMGenerationError,))
    def _call_simulate(
        self, problem: Problem, modified_problem: str, current_code: str
    ) -> str:
        prompt = self.prompt_manager.render_prompt(
            "initialization/dryrun/simulate.j2",
            question_content=modified_problem,
            current_code=current_code,
            language=self.language_name,
        )
        return self._generate(prompt)

    @llm_retry((LLMGenerationError,))
    def _call_refine_plan_with_sim(
        self, problem: Problem, modified_problem: str, plan: str, simulation_trace: str
    ) -> str:
        prompt = self.prompt_manager.render_prompt(
            "initialization/dryrun/plan_refine_with_sim.j2",
            question_content=modified_problem,
            starter_code=problem.starter_code,
            plan=plan,
            simulation_trace=simulation_trace,
        )
        return self._generate(prompt)

    @llm_retry((LLMGenerationError, LLMSyntaxError, ValueError))
    def _call_refine_code(
        self, problem: Problem, modified_problem: str, plan: str, current_code: str
    ) -> str:
        prompt = self.prompt_manager.render_prompt(
            "initialization/dryrun/code_refine.j2",
            question_content=modified_problem,
            plan=plan,
            starter_code=problem.starter_code,
            current_code=current_code,
            language=self.language_name,
        )
        response = self._generate(prompt)
        snip = self._extract_code_block(response)
        return self._validated_code(snip, problem.starter_code, "dryrun_refine")
