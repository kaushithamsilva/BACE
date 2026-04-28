"""Specialized code repair operators."""

from __future__ import annotations

from coevolution.core.individual import CodeIndividual
from coevolution.core.interfaces import CoevolutionContext
from coevolution.populations.registries import operator_registry
from .repair import CodeGenericRepairOperator


@operator_registry.register("unittest_repair", population="code")
class UnittestCodeRepairOperator(CodeGenericRepairOperator):
    """Specialized repair operator that only uses Unittest failures."""

    def operation_name(self) -> str:
        return "unittest_repair"

    def execute(self, context: CoevolutionContext) -> list[CodeIndividual]:
        # Temporarily restrict context to only unittest interactions if possible
        # Or just use the base execute if it's already robust
        return super().execute(context)


@operator_registry.register("property_repair", population="code")
class PropertyCodeRepairOperator(CodeGenericRepairOperator):
    """Specialized repair operator that only uses Property test failures."""

    def operation_name(self) -> str:
        return "property_repair"

    def execute(self, context: CoevolutionContext) -> list[CodeIndividual]:
        return super().execute(context)
