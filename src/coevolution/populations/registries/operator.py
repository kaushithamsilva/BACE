"""Registry for population genetic operators.

Allows operators to be registered with a name and population type,
enabling dynamic discovery and instantiation from YAML configuration.
"""

from __future__ import annotations
from typing import Any, Dict, List

from coevolution.core.interfaces.operators import IOperator, RegisteredOperator
from coevolution.strategies.breeding.breeder import Breeder
from .base import DIRegistry


class OperatorRegistry(DIRegistry[IOperator[Any]]):
    """Registry mapping (population, name) to operator classes."""

    def build_operator(
        self,
        name: str,
        population: str,
        **context: Any,
    ) -> Any:
        """Instantiate and return a single registered operator."""
        if population not in self._registry or name not in self._registry[population]:
            raise ValueError(f"Operator '{name}' not found for population '{population}'")

        cls = self._registry[population][name]
        return self._instantiate(cls, context)

    def build_weighted_breeder(
        self,
        population: str,
        config: Dict[str, Any],
        **dependencies: Any,
    ) -> Breeder[Any]:
        """Build a Breeder with weighted operators from YAML config.

        Args:
            population: The population type (e.g. 'code').
            config: Dict of YAML configuration keys (e.g. {mutation_rate: 0.2}).
            dependencies: Common dependencies (llm, parser, parent_selector, etc.)

        Returns:
            A Breeder instance.
        """
        registered_classes = self.get_all(population)
        if not registered_classes:
            raise ValueError(f"No operators registered for population '{population}'")

        # Combine config and explicit dependencies/overrides
        full_config = {**dependencies, **config}

        # 1. Map registered names to their configured weights
        weights: Dict[str, float] = {}
        for name in registered_classes:
            # Note: Operators use '{name}_rate' convention (e.g. mutation_rate)
            weight = full_config.get(f"{name}_rate", 0.0)
            if weight > 0:
                weights[name] = weight

        if not weights:
            # Fallback for single-operator populations or old configs
            # If only one operator is registered, give it 1.0 weight
            if len(registered_classes) == 1:
                name = list(registered_classes.keys())[0]
                weights[name] = 1.0
            else:
                raise ValueError(
                    f"No non-zero weights found in config for '{population}' operators. "
                    f"Available: {self.list_names(population)}"
                )

        # 2. Instantiate weighted operators
        registered_ops: List[RegisteredOperator[Any]] = []
        
        for name, weight in weights.items():
            cls = registered_classes[name]
            instance = self._instantiate(cls, full_config)
            registered_ops.append(RegisteredOperator(weight, instance))

        # 3. Create Breeder
        llm_workers = full_config.get("llm_workers")
        if llm_workers is None and "llm" in full_config:
            llm_workers = getattr(full_config["llm"], "workers", 1)
        
        return Breeder(registered_ops, llm_workers=llm_workers or 1)


# Global singleton
operator_registry = OperatorRegistry()
