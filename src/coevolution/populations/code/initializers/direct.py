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
from coevolution.strategies.llm_base import BaseLLMInitializer, ILanguageModel
from coevolution.populations.registries import initializer_registry
from ..operators._helpers import _CodeLLMHelpers


@initializer_registry.register("direct", population="code")
class DirectCodeInitializer(_CodeLLMHelpers, BaseLLMInitializer[CodeIndividual]):
    """Creates Gen-0 code individuals via parallel LLM calls (Zero-Shot)."""

    def __init__(
        self,
        llm: ILanguageModel,
        parser: ICodeParser,
        language_name: str,
        pop_config: PopulationConfig,
        llm_workers: int = 4,
    ) -> None:
        super().__init__(llm, parser, language_name, pop_config)
        self.llm_workers = llm_workers

    def initialize(self, problem: Problem, size: int | None = None) -> list[CodeIndividual]:
        target = size if size is not None else self.pop_config.initial_population_size
        if target <= 0:
            return []

        individuals: list[CodeIndividual] = []

        def _generate_single() -> str:
            prompt = self.prompt_manager.render_prompt(
                "operators/code/initial_single.j2",
                question_content=problem.question_content,
                starter_code=problem.starter_code,
            )
            response = self._generate(prompt)
            code = self._extract_code_block(response)
            code = self._validated_code(code, problem.starter_code, "initial")
            return code

        logger.info(
            f"DirectCodeInitializer: initializing {target} individuals using {self.llm_workers} threads"
        )
        with ThreadPoolExecutor(max_workers=self.llm_workers) as executor:
            futures = [executor.submit(_generate_single) for _ in range(target)]
            for future in as_completed(futures):
                try:
                    snip = future.result()
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
                    logger.error(f"Failed to generate individual: {e}")

        if not individuals:
            raise RuntimeError(
                "DirectCodeInitializer: failed to generate any individuals"
            )
        return individuals
