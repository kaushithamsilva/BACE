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

> [!TIP]
> If your operator shares complex logic with other components (like LLM-specific extraction or validation), place that logic in a dedicated `helpers/` folder within the population directory. This keeps the `operators/` folder clean and strictly limited to registered strategies.

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

The registry uses `inspect` to automatically inject dependencies into your constructor (`__init__`). For any operator inheriting from `BaseLLMOperator`, the following dependencies are **mandatory** and must be provided in the constructor:

- `llm`: The language model client (`ILanguageModel`).
- `parser`: The code parser for the current language (`ICodeParser`).
- `language_name`: The string name of the language (e.g., "python").
- `parent_selector`: The strategy used to select parents (`IParentSelectionStrategy`).
- `prob_assigner`: The strategy used to assign probabilities to offspring (`IProbabilityAssigner`).

Example constructor for a custom repair operator:

```python
def __init__(
    self,
    llm: ILanguageModel,
    parser: ICodeParser,
    language_name: str,
    parent_selector: IParentSelectionStrategy[Any],
    prob_assigner: IProbabilityAssigner,
    # Add custom parameters here (will be pulled from YAML)
    k_failing_tests: int = 10,
    **kwargs: Any,
) -> None:
    super().__init__(
        llm=llm,
        parser=parser,
        language_name=language_name,
        parent_selector=parent_selector,
        prob_assigner=prob_assigner,
    )
    self.k_failing_tests = k_failing_tests
```

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


### 4. Specialized Repair Operators (`*_repair`)

A unique feature of BACE is the **Specialized Repair** system. While most operators are "owned" by the population they evolve (e.g., a "mutation" operator for the `code` population), the `code` population often needs repair logic that is tightly coupled to the *test* population currently being co-evolved.

For example:
- **`unittest` population**: Uses `UnittestCodeRepairOperator` (registered as `code_repair` for `population="code"`).
- **`property` population**: Uses `PropertyCodeRepairOperator` (registered as `code_repair` for `population="code"`).
- **`public` population**: Uses `PublicCodeRepairOperator` (registered as `code_repair` for `population="code"`).

When the `PopulationDiscoveryService` builds the `code` population, it automatically searches for a `code_repair` operator. If your module exports a specialized version, the registry will prioritize it, allowing each environment to provide its own domain-specific repair strategy for the same target individuals.

---

## Summary Checklist

- [ ] Operator class implemented and decorated with `@operator_registry.register`.
- [ ] Class imported in `operators/__init__.py`.
- [ ] Rate assigned in `.yaml` config (e.g., `semantic_mutation_rate: 0.2`).
- [ ] Total rates in YAML sum to 1.0.

> [!NOTE]
> You do **not** need to modify the `profile.py` factory function for standard parameters. The registry handles discovery, instantiation, and weighting automatically.
