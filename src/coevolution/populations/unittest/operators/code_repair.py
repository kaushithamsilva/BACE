"""UnittestCodeRepairOperator — specialized code repair using unittest feedback."""

from __future__ import annotations

from coevolution.core.individual import CodeIndividual
from coevolution.core.interfaces import CoevolutionContext
from coevolution.populations.registries import operator_registry
from coevolution.populations.code.operators.repair import CodeGenericRepairOperator


@operator_registry.register("unittest_repair", population="code")
class UnittestCodeRepairOperator(CodeGenericRepairOperator):
    """Specialized repair operator that only uses Unittest failures."""

    def execute(self, context: CoevolutionContext) -> list[CodeIndividual]:
        """Repair using ONLY failing unittests."""
        # We temporarily decrease k if needed, but the main thing is the filter
        code_pop = context.code_population
        problem = context.problem

        parents = self.parent_selector.select_parents(code_pop, 1, context)
        if not parents:
            return []
        parent = parents[0]

        # Use the NEW filter capability
        failing = self._failing_test_selector.select_k_failing_tests(
            context, parent, k=self.k_failing_tests, test_type_filter="unittest"
        )
        if not failing:
            return []
        
        failing_tests_data = []
        for test_ind, test_pop_type in failing:
            exec_result = context.interactions[test_pop_type].execution_results
            trace = exec_result[parent.id][test_ind.id].error_log or "No trace available"
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
            self.operation_name(), [parent.probability]
        )
        return [
            CodeIndividual(
                snippet=edited_code,
                probability=probability,
                creation_op=self.operation_name(),
                generation_born=code_pop.generation + 1,
                parents={"code": [parent.id], "test": [t.id for t, _ in failing]},
                explanation=self.parser.get_docstring(edited_code),
                metadata={"num_failing_tests": len(failing), "target_test_type": "unittest"},
            )
        ]
