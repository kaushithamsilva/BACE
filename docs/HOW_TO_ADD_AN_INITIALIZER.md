# Adding a New Initializer

This guide explains how to add a new initialization strategy (e.g., a new planning-based approach or a zero-shot template) to a population.

The process is "zero-friction" and requires only two steps:
1. Implement the class and tag with `@initializer_registry.register`.
2. Assign the weight in your YAML configuration.

---

## 1. Write the implementation

Create your new initializer inside `populations/<name>/operators/<new_initializer>.py`.

It should inherit from `BaseCodeInitializer` (for code populations) or `BaseLLMInitializer[TestIndividual]` (for test populations).

```python
# coevolution/populations/code/operators/zero_shot.py
from coevolution.populations.code.operators.initializer import BaseCodeInitializer
from coevolution.populations.initializer_registry import initializer_registry

@initializer_registry.register("zero_shot", population="code")
class ZeroShotCodeInitializer(BaseCodeInitializer):
    def initialize(self, problem: Problem, size: int | None = None) -> list[CodeIndividual]:
        # Your logic here...
        # respect the 'size' parameter!
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

## 2. Export It (1 line)

Make the initializer available so the decorator runs by exporting it in `populations/<name>/operators/__init__.py`:

```python
from .zero_shot import ZeroShotCodeInitializer

__all__ = [
    # ...
    "ZeroShotCodeInitializer",
]
```

## 3. Update the YAML Config

Enable your new initializer by setting its rate in your experiment YAML. The key is always `{name}_init_rate`.

```yaml
# configs/profiles/code/experiment.yaml
code_profile:
  zero_shot_init_rate: 0.5
  standard_init_rate: 0.5
```

---

## Summary Checklist
- [ ] Initializer class implemented and decorated with `@initializer_registry.register`.
- [ ] Class imported in `operators/__init__.py`.
- [ ] Weight assigned in `.yaml` config (e.g., `my_new_init_rate: 1.0`).

> [!NOTE]
> You do **not** need to modify `profile.py` or the `create_<name>_profile` factory function. The registry handles discovery and instantiation automatically.
