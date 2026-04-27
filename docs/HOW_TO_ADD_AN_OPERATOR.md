# Adding a New Genetic Operator

This guide explains how to add a new evolutionary operator (e.g., a new heuristic mutation, a novel crossover technique) to a population using the **Operator Registry**.

The process is "zero-friction" and requires only three steps:
1. Implement the class and tag with `@operator_registry.register`.
2. Export it in the package `__init__.py`.
3. Assign the weight in your YAML configuration.

---

## 1. Write the Implementation

Create your new operator inside `src/coevolution/populations/<name>/operators/<new_operation>.py`.

It should inherit from `BaseLLMOperator` (for LLM-based operators) or implement the `IOperator` protocol.

```python
# src/coevolution/populations/code/operators/semantic.py
from coevolution.strategies.llm_base import BaseLLMOperator, LLMGenerationError, llm_retry
from coevolution.core.individual import CodeIndividual
from coevolution.core.interfaces import CoevolutionContext
from coevolution.populations.registries import operator_registry

@operator_registry.register("semantic_mutation", population="code")
class SemanticHeuristicOperator(BaseLLMOperator[CodeIndividual]):
    def operation_name(self) -> str:
        return "semantic_mutation"

    @llm_retry((ValueError, LLMGenerationError))
    def execute(self, context: CoevolutionContext) -> list[CodeIndividual]:
        # Your logic here...
        return []
```

### Dependency Injection
The registry uses `inspect` to automatically inject dependencies into your constructor (`__init__`). You can request:
- `llm`: The language model client.
- `parser`: The code parser for the current language.
- `language_name`: The string name of the language (e.g., "python").
- `parent_selector`: The strategy used to select parents.
- `prob_assigner`: The strategy used to assign probabilities to offspring.
- Any other parameter passed to `build_weighted_breeder` in the profile factory.

## 2. Export It (1 line)

Make the operator available so the decorator runs during system initialization. Add it to `src/coevolution/populations/<name>/operators/__init__.py`:

```python
from .semantic import SemanticHeuristicOperator

__all__ = [
    # ...
    "SemanticHeuristicOperator",
]
```

## 3. Update the YAML Config

Enable your new operator by setting its rate in your experiment YAML. The key convention is `{name}_rate`.

```yaml
# configs/profiles/code/experiment.yaml
code_profile:
  semantic_mutation_rate: 0.2
  mutation_rate: 0.2
  crossover_rate: 0.2
  generic_edit_rate: 0.4
```

> [!IMPORTANT]
> The sum of all `*_rate` values for a population must equal exactly **1.0**.

---

## Summary Checklist
- [ ] Operator class implemented and decorated with `@operator_registry.register`.
- [ ] Class imported in `operators/__init__.py`.
- [ ] Rate assigned in `.yaml` config (e.g., `semantic_mutation_rate: 0.2`).
- [ ] Total rates in YAML sum to 1.0.

> [!NOTE]
> You do **not** need to modify the `profile.py` factory function for standard parameters. The registry handles discovery, instantiation, and weighting automatically.
