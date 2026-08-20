from collections.abc import Callable, Sequence

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.tool_execution import ToolsExecutor, tool_result_to_str
from agentdojo.defenses.task_shield.core import TaskShield
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionReturnType, FunctionsRuntime, TaskEnvironment
from agentdojo.types import (
    ChatAssistantMessage,
    ChatMessage,
    ChatToolResultMessage,
    ChatUserMessage,
    FunctionCall,
    get_text_content_as_str,
    text_content_block_from_string,
)

BLOCKED_TOOL_CALL_ERROR = "Blocked by Task Shield because the tool call is not aligned with the user task."


def render_history(messages: Sequence[ChatMessage]) -> str:
    rendered = []
    for message in messages:
        content = get_text_content_as_str(message["content"] or [])
        if message["role"] == "assistant" and message.get("tool_calls"):
            calls = ", ".join(f"{call.function}({dict(call.args)})" for call in message["tool_calls"] or [])
            content = f"{content}\ntool_calls: {calls}" if content else f"tool_calls: {calls}"
        rendered.append(f"{message['role']}: {content}")
    return "\n".join(rendered)


class TaskShieldUserTaskExtractor(BasePipelineElement):
    def __init__(self, shield: TaskShield) -> None:
        self.shield = shield

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        self.shield.reset(query)
        return query, runtime, env, messages, extra_args


class TaskShieldLLM(BasePipelineElement):
    """Checks assistant natural-language content and asks the agent to rethink it."""

    def __init__(self, llm: BasePipelineElement, shield: TaskShield, max_feedback_rounds: int = 2) -> None:
        self.llm = llm
        self.shield = shield
        self.max_feedback_rounds = max_feedback_rounds
        self.name = llm.name

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        current_messages = messages
        feedback = "Task Shield blocked a persistently misaligned response."
        for _ in range(self.max_feedback_rounds + 1):
            query, runtime, env, generated, extra_args = self.llm.query(
                query, runtime, env, current_messages, extra_args
            )
            assistant = generated[-1]
            if assistant["role"] != "assistant":
                return query, runtime, env, generated, extra_args
            content = get_text_content_as_str(assistant["content"] or [])
            instructions = self.shield.extract(content, "the assistant")
            history = render_history(generated[:-1])
            misaligned = [
                instruction
                for instruction in instructions
                if not self.shield.instruction_is_aligned(instruction, "assistant", history)
            ]
            if not misaligned:
                return query, runtime, env, generated, extra_args

            feedback = self.shield.feedback(misaligned)
            safe_assistant = ChatAssistantMessage(
                role="assistant",
                content=assistant["content"],
                tool_calls=None,
            )
            current_messages = [
                *generated[:-1],
                safe_assistant,
                ChatUserMessage(role="user", content=[text_content_block_from_string(feedback)]),
            ]

        blocked = ChatAssistantMessage(
            role="assistant",
            content=[text_content_block_from_string(feedback)],
            tool_calls=None,
        )
        return query, runtime, env, [*current_messages, blocked], extra_args


class TaskShieldToolsExecutor(ToolsExecutor):
    """Blocks misaligned tool calls and annotates misaligned tool outputs."""

    def __init__(
        self,
        shield: TaskShield,
        tool_output_formatter: Callable[[FunctionReturnType], str] = tool_result_to_str,
    ) -> None:
        super().__init__(tool_output_formatter)
        self.shield = shield

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        if not messages or messages[-1]["role"] != "assistant" or not messages[-1].get("tool_calls"):
            return query, runtime, env, messages, extra_args

        assistant = messages[-1]
        tool_calls = assistant["tool_calls"] or []
        related_content = get_text_content_as_str(assistant["content"] or [])
        history = render_history(messages[:-1])
        blocked = {
            index
            for index, call in enumerate(tool_calls)
            if not self.shield.tool_call_is_aligned(call.function, call.args, related_content, history)
        }
        allowed_calls = [call for index, call in enumerate(tool_calls) if index not in blocked]
        executed_results = self._execute_allowed(query, runtime, env, messages, extra_args, allowed_calls)
        executed_iter = iter(executed_results)
        results: list[ChatToolResultMessage] = []
        for index, call in enumerate(tool_calls):
            if index in blocked:
                instruction = f"Function: {call.function} Arguments: {dict(call.args)}"
                results.append(
                    ChatToolResultMessage(
                        role="tool",
                        content=[text_content_block_from_string(self.shield.feedback([instruction]))],
                        tool_call_id=call.id,
                        tool_call=call,
                        error=BLOCKED_TOOL_CALL_ERROR,
                    )
                )
            else:
                result = next(executed_iter)
                self._check_tool_output(result, history)
                results.append(result)
        return query, runtime, env, [*messages, *results], extra_args

    def _execute_allowed(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: TaskEnvironment,
        messages: Sequence[ChatMessage],
        extra_args: dict,
        calls: Sequence[FunctionCall],
    ) -> list[ChatToolResultMessage]:
        if not calls:
            return []
        assistant = ChatAssistantMessage(
            role="assistant",
            content=messages[-1]["content"],
            tool_calls=list(calls),
        )
        _, _, _, executed, _ = super().query(query, runtime, env, [*messages[:-1], assistant], extra_args)
        return list(executed[len(messages) :])  # type: ignore[return-value]

    def _check_tool_output(self, result: ChatToolResultMessage, history: str) -> None:
        output = get_text_content_as_str(result["content"] or [])
        if not output.strip():
            return
        call = result["tool_call"]
        source = f"from tool [{call.function}] with arguments [{dict(call.args)}]"
        instructions = self.shield.extract(output, f"the tool {call.function}")
        misaligned = [
            instruction
            for instruction in instructions
            if not self.shield.instruction_is_aligned(instruction, "tool", history, source)
        ]
        if misaligned:
            result["content"] = [
                *(result["content"] or []),
                text_content_block_from_string(self.shield.feedback(misaligned, result["tool_call_id"])),
            ]
