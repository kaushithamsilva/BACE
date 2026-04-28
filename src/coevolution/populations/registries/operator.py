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

    def _extract_rates(
        self, population: str, config: Dict[str, Any], section_name: str = "op_rates"
    ) -> Dict[str, float]:
        """Extract and validate sampling rates from a named config section."""
        rates = config.get(section_name)
        if rates is None:
            raise ValueError(
                f"Mandatory section '{section_name}' is missing for population '{population}'. "
                f"Please update your YAML config."
            )

        registered_names = self.list_names(population)
        
        # 1. Validate that all operators listed in YAML are actually registered
        for name in rates.keys():
            if name not in registered_names:
                raise ValueError(
                    f"Operator '{name}' listed in '{section_name}' is not registered for "
                    f"population '{population}'. Registered: {registered_names}"
                )

        # 2. Extract weights (only for those listed in the section)
        weights = {name: float(weight) for name, weight in rates.items() if weight > 0}
        
        if not weights:
            raise ValueError(
                f"No non-zero weights found in '{section_name}' for '{population}'. "
                f"At least one operator must be enabled."
            )

        # 3. Strict Sum-to-1.0 Validation
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Operator weights in '{section_name}' for '{population}' must sum to 1.0, "
                f"got {total:.4f} (from {weights})"
            )

        return weights

    def build_weighted_breeder(
        self,
        population: str,
        config: Dict[str, Any],
        **dependencies: Any,
    ) -> Breeder[Any]:
        """Build a Breeder with weighted operators from YAML config.

        Args:
            population: The population type (e.g. 'code').
            config: Dict of YAML configuration keys (must contain 'op_rates').
            dependencies: Common dependencies (llm, parser, parent_selector, etc.)

        Returns:
            A Breeder instance.
        """
        # 1. Extract and validate rates
        weights = self._extract_rates(population, config)

        # 2. Combine config and explicit dependencies for instantiation
        full_config = {**dependencies, **config}

        # 3. Instantiate weighted operators
        registered_classes = self.get_all(population)
        registered_ops: List[RegisteredOperator[Any]] = []
        
        for name, weight in weights.items():
            cls = registered_classes[name]
            instance = self._instantiate(cls, full_config)
            registered_ops.append(RegisteredOperator(weight, instance))

        # 4. Create Breeder
        llm_workers = full_config.get("llm_workers")
        if llm_workers is None and "llm" in full_config:
            llm_workers = getattr(full_config["llm"], "workers", 1)
        
        return Breeder(registered_ops, llm_workers=llm_workers or 1)


# Global singleton
operator_registry = OperatorRegistry()
