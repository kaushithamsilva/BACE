"""AgentCoderInitializer — Turn 0 of the AgentCoder loop."""

from __future__ import annotations


from coevolution.core.individual import CodeIndividual
from coevolution.core.interfaces import (
    OPERATION_INITIAL,
    Problem,
)
from coevolution.populations.registries import initializer_registry
from coevolution.core.interfaces.language import (
    ICodeParser,
    LanguageParsingError,
    LanguageTransformationError,
)

from coevolution.strategies.llm_base import (
    BaseLLMInitializer,
    ILanguageModel,
    LLMGenerationError,
    LLMSyntaxError,
    llm_retry,
)
from ..operators.repair import AgentCoderRepairOperator


@initializer_registry.register("agent_coder", population="agent_coder")
class AgentCoderInitializer(BaseLLMInitializer[CodeIndividual]):
    """Turn 0 of the AgentCoder loop: generates the first solution and seeds history.

    MUST be called before AgentCoderRepairOperator.execute().
    Pass the same operator instance to both so they share conversation history.
    """

    def __init__(
        self,
        llm: ILanguageModel,
        parser: ICodeParser,
        language_name: str,
        edit_operator: AgentCoderRepairOperator,
    ) -> None:
        super().__init__(llm, parser, language_name)
        self._edit_operator = edit_operator

    def initialize(self, problem: Problem, size: int | None = None) -> list[CodeIndividual]:
        if size is not None and size != 1:
             raise ValueError("AgentCoder initializer only supports size=1 per session.")
        self._edit_operator.reset_session()
        return [self._generate_initial(problem)]

    @llm_retry(
        (
            ValueError,
            LanguageParsingError,
            LanguageTransformationError,
            LLMGenerationError,
            LLMSyntaxError,
        )
    )
    def _generate_initial(self, problem: Problem) -> CodeIndividual:
        prompt = self.prompt_manager.render_prompt(
            "operators/agent_coder/init.j2",
            question_content=problem.question_content,
            starter_code=problem.starter_code,
        )
        self._edit_operator._conversation_history.append(
            {"role": "user", "content": prompt}
        )
        response = self._generate(self._edit_operator._conversation_history)
        self._edit_operator._conversation_history.append(
            {"role": "assistant", "content": response}
        )

        code_blocks = self.parser.extract_code_blocks(response)
        if not code_blocks:
            raise ValueError("AgentCoderInitializer: no code block in LLM response")
        code = code_blocks[-1]
        self._validate_syntax(code)

        if not self.parser.contains_starter_code(code, problem.starter_code):
            raise ValueError(
                "AgentCoderInitializer: generated code missing starter code"
            )

        code = self.parser.remove_main_block(code)

        return CodeIndividual(
            snippet=code,
            probability=0.0,  # Set by orchestrator
            creation_op=OPERATION_INITIAL,
            generation_born=0,
            explanation=self.parser.get_docstring(code),
        )


__all__ = ["AgentCoderInitializer"]
