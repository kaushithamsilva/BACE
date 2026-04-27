"""Registry for population genetic operators.

Allows operators to be registered with a name and population type,
enabling dynamic discovery and instantiation from YAML configuration.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Type, TypeVar

from loguru import logger

from coevolution.core.interfaces.base import BaseIndividual
from coevolution.core.interfaces.operators import IOperator, RegisteredOperator
from coevolution.strategies.breeding.breeder import Breeder

T = TypeVar("T", bound=BaseIndividual)

class OperatorRegistry:
    """Registry mapping (population, name) to operator classes."""

    def __init__(self) -> None:
        # Structure: {population_type: {name: class}}
        self._registry: Dict[str, Dict[str, Type[IOperator[Any]]]] = {}

    def register(self, name: str, population: str) -> Callable[[Type[IOperator[Any]]], Type[IOperator[Any]]]:
        """Decorator to register an operator class.

        Args:
            name: Unique name for this operator within the population (e.g. 'mutation').
            population: Population type (e.g. 'code', 'unittest').
        """
        def wrapper(cls: Type[IOperator[Any]]) -> Type[IOperator[Any]]:
            if population not in self._registry:
                self._registry[population] = {}
            self._registry[population][name] = cls
            logger.trace(f"Registered operator '{name}' for population '{population}'")
            return cls
        return wrapper

    def get_all(self, population: str) -> Dict[str, Type[IOperator[Any]]]:
        """Get all registered operators for a population."""
        return self._registry.get(population, {}).copy()

    def list_names(self, population: str) -> List[str]:
        """List registered names for a population."""
        return list(self._registry.get(population, {}).keys())

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

        # 1. Map registered names to their configured weights
        weights: Dict[str, float] = {}
        for name in registered_classes:
            # Note: Operators use '{name}_rate' convention (e.g. mutation_rate)
            weight = config.get(f"{name}_rate", 0.0)
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
        context = {**dependencies, **config}
        registered_ops: List[RegisteredOperator[Any]] = []
        
        for name, weight in weights.items():
            cls = registered_classes[name]
            instance = self._instantiate(cls, context)
            registered_ops.append(RegisteredOperator(weight, instance))

        # 3. Create Breeder
        llm_workers = 1
        if "llm" in dependencies:
            llm_workers = getattr(dependencies["llm"], "workers", 1)
        
        return Breeder(registered_ops, llm_workers=llm_workers)

    def _instantiate(self, cls: Type[IOperator[Any]], context: Dict[str, Any]) -> IOperator[Any]:
        """Instantiate a class by injecting matching parameters from the context."""
        sig = inspect.signature(cls.__init__)
        
        # Inject context values that match parameter names
        kwargs = {
            k: v for k, v in context.items()
            if k in sig.parameters
        }
        
        # Check for missing required parameters (those without a default value)
        for name, param in sig.parameters.items():
            if name in {"self", "args", "kwargs"}:
                continue
            if name not in kwargs and param.default is inspect.Parameter.empty:
                logger.warning(
                    f"Required parameter '{name}' for {cls.__name__} not found in creation context."
                )
                
        return cls(**kwargs)

# Global singleton
operator_registry = OperatorRegistry()
