# Adding a New Initializer

This guide explains how to add a new initialization strategy (e.g., a new planning-based approach or a zero-shot template) to a population.

The process is "zero-friction" and requires only two steps:

1. Implement the class and tag with `@initializer_registry.register`.
2. Assign the weight in your YAML configuration.

> [!NOTE]
> The system uses automatic module discovery. You **do not** need to manually import your new initializer in `__init__.py`. As long as your file is inside the `initializers/` directory and contains the `@register` decorator at the class level, it will be automatically discovered at runtime.

---

## 1. Write the implementation

Create your new initializer inside `src/coevolution/populations/<name>/initializers/<new_initializer>.py`.

For code populations, inherit from `CodeLLMHelpers` (for utility methods) and `BaseLLMInitializer[CodeIndividual]`. For other populations, inherit from `BaseLLMInitializer[T]`.

```python
# src/coevolution/populations/code/initializers/zero_shot.py
from coevolution.populations.code.helpers.llm_helpers import CodeLLMHelpers
from coevolution.strategies.llm_base import BaseLLMInitializer
from coevolution.populations.registries import initializer_registry

@initializer_registry.register("zero_shot", population="code")
class ZeroShotCodeInitializer(CodeLLMHelpers, BaseLLMInitializer[CodeIndividual]):
    def initializer_name(self) -> str:
        return "zero_shot"

    def initialize(self, problem: Problem, size: int | None = None) -> list[CodeIndividual]:
        # Your logic here...
        # Note: All orchestrations like parallelism are handled by WeightedPopulationInitializer.
        return []
```

### Dependency Injection

The registry uses `inspect` to automatically inject dependencies into your constructor. You can ask for:

- `llm`: The language model client.
- `parser`: The code parser for the current language.
- `language_name`: The string name of the language (e.g., "python").
- `pop_config`: The `PopulationConfig` instance.
- `sandbox_config`: (If applicable) The sandbox configuration.
- Any other parameter passed to `build_weighted_initializer` in the profile factory.

## 2. Automatic Discovery

The system automatically discovers all initializers in the `initializers/` directory. There is **no need** to manually import or export your new initializer in `src/coevolution/populations/<name>/initializers/__init__.py`.

## 3. Update the YAML Config

Enable your new initializer by setting its rate in your experiment YAML. The key is always `{name}_init_rate`.

```yaml
# configs/profiles/code/experiment.yaml
code_profile:
  zero_shot_init_rate: 0.5
  direct_init_rate: 0.5
```

---

## Parallel Execution

- **Batching is removed**: All initializers should generate individuals 1-by-1 in parallel using a `ThreadPoolExecutor` internally (e.g., see `DirectCodeInitializer`).
- **Cross-initializer concurrency**: The `WeightedPopulationInitializer` triggers all weighted sub-initializers in parallel. This means multiple strategies start working concurrently.

---

## Summary Checklist

- [ ] Initializer class implemented and decorated with `@initializer_registry.register`.
- [ ] Weight assigned in `.yaml` config (e.g., `my_new_init_rate: 1.0`).

> [!NOTE]
> You do **not** need to modify `profile.py` or the `create_<name>_profile` factory function. The registry handles discovery and instantiation automatically.
