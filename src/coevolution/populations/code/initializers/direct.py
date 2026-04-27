"""Direct Code Initializer."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger

from coevolution.core.individual import CodeIndividual
from coevolution.core.interfaces import (
    OPERATION_INITIAL,
    PopulationConfig,
    Problem,
)
from coevolution.core.interfaces.language import ICodeParser
from coevolution.strategies.llm_base import ILanguageModel
from coevolution.populations.registries import initializer_registry
from .base import BaseCodeInitializer


@initializer_registry.register("direct", population="code")
class DirectCodeInitializer(BaseCodeInitializer):
    """Creates Gen-0 code individuals via batched LLM calls (Zero-Shot)."""

    def __init__(
        self,
        llm: ILanguageModel,
        parser: ICodeParser,
        language_name: str,
        pop_config: PopulationConfig,
        init_pop_batch_size: int = 2,
        llm_workers: int = 4,
    ) -> None:
        super().__init__(llm, parser, language_name, pop_config, llm_workers)
        self.init_pop_batch_size = min(
            init_pop_batch_size, pop_config.initial_population_size or 1
        )

    def initialize(self, problem: Problem, size: int | None = None) -> list[CodeIndividual]:
        target = size if size is not None else self.pop_config.initial_population_size
        # Clamp batch size to target in case the caller requested fewer individuals
        # than the default batch size configured at construction time.
        effective_batch = min(self.init_pop_batch_size, target) if target > 0 else 1
        individuals: list[CodeIndividual] = []
        num_batches = (target + effective_batch - 1) // effective_batch if target > 0 else 0

        def _generate_batch(batch_size: int) -> list[str]:
            if batch_size == 1:
                prompt = self.prompt_manager.render_prompt(
                    "operators/code/initial_single.j2",
                    question_content=problem.question_content,
                    starter_code=problem.starter_code,
                )
                response = self._generate(prompt)
                code = self._extract_code_block(response)
                code = self._validated_code(code, problem.starter_code, "initial")
                return [code]
            else:
                prompt = self.prompt_manager.render_prompt(
                    "operators/code/initial_population.j2",
                    question_content=problem.question_content,
                    starter_code=problem.starter_code,
                    population_size=batch_size,
                )
                response = self._generate(prompt)
                blocks = self._extract_all_code_blocks(response)
                validated_blocks = []
                for b in blocks:
                    validated_blocks.append(
                        self._validated_code(b, problem.starter_code, "initial")
                    )
                if len(validated_blocks) != batch_size:
                    raise ValueError(
                        f"Expected {batch_size} code blocks, got {len(validated_blocks)}"
                    )
                return validated_blocks

        logger.info(
            f"DirectCodeInitializer: initializing {target} individuals in {num_batches} batches"
        )
        with ThreadPoolExecutor(max_workers=self.llm_workers) as executor:
            futures = [
                executor.submit(_generate_batch, effective_batch)
                for _ in range(num_batches)
            ]
            for future in as_completed(futures):
                try:
                    snippets = future.result()
                    for snip in snippets:
                        individuals.append(
                            CodeIndividual(
                                snippet=snip,
                                probability=self.pop_config.initial_prior,
                                creation_op=OPERATION_INITIAL,
                                generation_born=0,
                                explanation=self.parser.get_docstring(snip),
                                metadata={"initializer": self.__class__.__name__},
                            )
                        )
                except Exception as e:
                    logger.error(f"Batch init failed: {e}")

        if not individuals:
            raise RuntimeError(
                "DirectCodeInitializer: failed to generate any individuals"
            )
        return individuals[:target]
