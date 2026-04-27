"""Registry for population initializers.

Allows initializers to be registered with a name and population type,
enabling dynamic discovery and instantiation from YAML configuration.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Type, TypeVar

from loguru import logger

from coevolution.core.interfaces.initializer import (
    IPopulationInitializer,
    RegisteredInitializer,
    WeightedPopulationInitializer,
)

T = TypeVar("T", bound=IPopulationInitializer[Any])

class InitializerRegistry:
    """Registry mapping (population, name) to initializer classes."""

    def __init__(self) -> None:
        # Structure: {population_type: {name: class}}
        self._registry: Dict[str, Dict[str, Type[IPopulationInitializer[Any]]]] = {}

    def register(self, name: str, population: str) -> Callable[[Type[IPopulationInitializer[Any]]], Type[IPopulationInitializer[Any]]]:
        """Decorator to register an initializer class.

        Args:
            name: Unique name for this initializer within the population (e.g. 'standard').
            population: Population type (e.g. 'code', 'unittest').
        """
        def wrapper(cls: Type[IPopulationInitializer[Any]]) -> Type[IPopulationInitializer[Any]]:
            if population not in self._registry:
                self._registry[population] = {}
            self._registry[population][name] = cls
            logger.trace(f"Registered initializer '{name}' for population '{population}'")
            return cls
        return wrapper

    def get_all(self, population: str) -> Dict[str, Type[IPopulationInitializer[Any]]]:
        """Get all registered initializers for a population."""
        return self._registry.get(population, {}).copy()

    def list_names(self, population: str) -> List[str]:
        """List registered names for a population."""
        return list(self._registry.get(population, {}).keys())

    def build_weighted_initializer(
        self,
        population: str,
        config: Dict[str, Any],
        **dependencies: Any,
    ) -> WeightedPopulationInitializer[Any]:
        """Build a WeightedPopulationInitializer from YAML config.

        Args:
            population: The population type (e.g. 'code').
            config: Dict of YAML configuration keys (e.g. {standard_init_rate: 1.0}).
            dependencies: Common dependencies (llm, parser, pop_config, etc.)

        Returns:
            A WeightedPopulationInitializer instance.
        """
        registered_classes = self.get_all(population)
        if not registered_classes:
            raise ValueError(f"No initializers registered for population '{population}'")

        # Required by WeightedPopulationInitializer itself
        pop_config = dependencies.get("pop_config")
        if not pop_config:
            raise ValueError(f"Dependency 'pop_config' required for population '{population}'")

        # 1. Map registered names to their configured weights
        weights: Dict[str, float] = {}
        for name in registered_classes:
            weight = config.get(f"{name}_init_rate", 0.0)
            if weight > 0:
                weights[name] = weight

        # 2. Backward compatibility fallback: use first registered if all weights are 0
        if not weights:
            default_name = sorted(registered_classes.keys())[0]
            logger.warning(
                f"No weights found in config for '{population}' initializers. "
                f"Falling back to '{default_name}' (weight=1.0)"
            )
            weights[default_name] = 1.0

        # 3. Instantiate those with non-zero weights
        context = {**dependencies, **config}
        registered_inits: List[RegisteredInitializer[Any]] = []
        
        for name, weight in weights.items():
            cls = registered_classes[name]
            instance = self._instantiate(cls, context)
            registered_inits.append(RegisteredInitializer(weight, instance))

        return WeightedPopulationInitializer(registered_inits, pop_config)

    def _instantiate(self, cls: Type[IPopulationInitializer[Any]], context: Dict[str, Any]) -> IPopulationInitializer[Any]:
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
initializer_registry = InitializerRegistry()
