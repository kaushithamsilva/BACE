# Adding a New Population

Each population lives in its own subfolder under `src/coevolution/populations/`. This guide explains the structure and registration process for a new population type.

---

## Folder Contract

Every population follows this structure:

```
populations/<name>/
├── __init__.py          # re-exports factory
├── profile.py           # factory function: create_<name>_profile(...)
└── operators/
    ├── __init__.py      # re-exports all operators (to trigger registry)
    ├── _helpers.py      # optional: private LLM utility mixin
    ├── mutation.py      # optional: <Name>MutationOperator
    ├── crossover.py     # optional: <Name>CrossoverOperator
    ├── edit.py          # optional: <Name>EditOperator
    └── initializer.py  # required: <Name>Initializer
```

---

## Step-by-Step

### 1. Create the folder structure
`src/coevolution/populations/<name>/`

### 2. Implement and Register Operators
Each operator and initializer should use the central registries for discovery and dependency injection.

#### Operator Example:
```python
# populations/<name>/operators/mutation.py
from coevolution.populations.registries import operator_registry

@operator_registry.register("mutation", population="<name>")
class <Name>MutationOperator(BaseLLMOperator[<Individual>]):
    ...
```

#### Initializer Example:
```python
# populations/<name>/operators/initializer.py
from coevolution.populations.registries import initializer_registry

@initializer_registry.register("standard", population="<name>")
class <Name>Initializer(BaseLLMInitializer[<Individual>]):
    ...
```

### 3. Write the profile factory
The factory function is responsible for high-level population configuration and delegating strategy construction to the registries.

```python
# populations/<name>/profile.py
from coevolution.core.interfaces import CodeProfile, PopulationConfig
from coevolution.populations.registries import (
    profile_registry, 
    initializer_registry, 
    operator_registry
)

@profile_registry.code_factory("<name>")  # or @profile_registry.test_factory("<name>")
def create_<name>_profile(llm_client, language_adapter, **factory_config) -> CodeProfile:
    pop_config = PopulationConfig(...)
    
    # Delegate construction to registries
    initializer = initializer_registry.build_weighted_initializer(
        population="<name>",
        config=factory_config,
        llm=llm_client,
        pop_config=pop_config
    )
    
    breeder = operator_registry.build_weighted_breeder(
        population="<name>",
        config=factory_config,
        llm=llm_client,
        parent_selector=...
    )
    
    return CodeProfile(
        population_config=pop_config,
        initializer=initializer,
        breeder=breeder
    )
```

### 4. Wire everything up
Ensure all operator files are imported in `operators/__init__.py` so their decorators run. Re-export the factory in the main population `__init__.py`.

### 5. Register in Global Populations
Add `from . import <name>` to `src/coevolution/populations/__init__.py`.

---

## Checklist
- [ ] Subfolder created in `src/coevolution/populations/`
- [ ] Initializer/Operators implement their logic and use `@*_registry.register`
- [ ] `operators/__init__.py` imports all operator implementations
- [ ] `profile.py` uses registries to build `initializer` and `breeder`
- [ ] `src/coevolution/populations/__init__.py` imports the new package
- [ ] Config rates (e.g. `standard_init_rate: 1.0`, `mutation_rate: 1.0`) added to experiment YAML.

> [!NOTE]
> You do **not** need to modify the `profile.py` factory function for standard parameters. The registry handles discovery, instantiation, and weighting automatically.
