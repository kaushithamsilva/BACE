"""UnittestCodeRepairOperator — specialized code repair using unittest feedback."""

from __future__ import annotations

from typing import Any

from coevolution.core.interfaces.language import ICodeParser
from coevolution.strategies.llm_base import ILanguageModel
from coevolution.core.interfaces.probability import IProbabilityAssigner
from coevolution.core.interfaces.selection import IParentSelectionStrategy
from coevolution.populations.code.operators.repair import CodeGenericRepairOperator
from coevolution.populations.registries import operator_registry
from coevolution.strategies.selection.failing_test_selection import FailingTestSelector


@operator_registry.register("unittest_repair", population="code")
class UnittestCodeRepairOperator(CodeGenericRepairOperator):
    """Specialized wrapper that fixes 'target_test_type' to 'unittest'."""

    def __init__(
        self,
        llm: ILanguageModel,
        parser: ICodeParser,
        language_name: str,
        parent_selector: IParentSelectionStrategy,
        prob_assigner: IProbabilityAssigner,
        failing_test_selector: type[FailingTestSelector] = FailingTestSelector,
        k_failing_tests: int = 10,
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
            target_test_type="unittest",
        )

    def operation_name(self) -> str:
        return "unittest_repair"
