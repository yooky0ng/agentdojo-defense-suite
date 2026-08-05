"""Official IPIGuard DAG execution classes from commit 4e686ed2."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence

from networkx import topological_sort

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.ipiguard.compatibility import ChatAssistantMessage, ChatToolResultMessage
from agentdojo.agent_pipeline.ipiguard.ipiguard_llm import OpenAITraverseLLM
from agentdojo.agent_pipeline.ipiguard.tool_white_list import whitelist
from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionCall, FunctionReturnType, FunctionsRuntime
from agentdojo.types import ChatMessage

class DagToolsExecutionLoop(BasePipelineElement):
    """Executes in loop a sequence of pipeline elements related to tool execution until the
    LLM does not return any tool calls.

    Args:
        elements: a sequence of pipeline elements to be executed in loop. One of them should be
            an LLM, and one of them should be a [ToolsExecutor][agentdojo.agent_pipeline.ToolsExecutor] (or
            something that behaves similarly by executing function calls). You can find an example usage
            of this class [here](../../concepts/agent_pipeline.md#combining-pipeline-components).
        max_iters: maximum number of iterations to execute the pipeline elements in loop.
    """

    def __init__(self, executor, max_iters: int = 15) -> None:
        self.executor = executor
        self.max_iters = max_iters

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        if len(messages) == 0:
            raise ValueError("Messages should not be empty when calling ToolsExecutionLoop")

        dag = extra_args['dag']

        for node in topological_sort(dag):
            tool_call = dag.nodes[node]

            extra_args["current_node"] = node
            extra_args["current_tool_call"] = tool_call["function_call"]
            
            query, runtime, env, messages, extra_args = self.executor.query(query, runtime, env, messages, extra_args)  

        # final output
        query, runtime, env, messages, extra_args = self.executor.traverse_llm.query_response(query, runtime, env, messages, extra_args)
        return query, runtime, env, messages, extra_args

class DagToolsExecutor(BasePipelineElement):
    """Executes the tool calls in the last messages for which tool execution is required.

    Args:
        tool_output_formatter: a function that converts a tool's output into plain text to be fed to the model.
            It should take as argument the tool output, and convert it into a string. The default converter
            converts the output to structured YAML.
    """

    def __init__(self, traverse_llm, tool_output_formatter: Callable[[FunctionReturnType], str] = tool_result_to_str) -> None:
        self.traverse_llm = traverse_llm
        self.output_formatter = tool_output_formatter

    def _run_tool_call_with_reflection(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]: 
        # run with reflection
        tool_call = extra_args["current_tool_call"]
        tool_call_result, error = runtime.run_function(env, tool_call.function, tool_call.args)
        extra_args["error_messages"] = []
        for _ in range(3):
            if error is None:
                break
            extra_args["error_messages"].append((tool_call, error))
            query, runtime, env, messages, extra_args = self.traverse_llm.query_reflection(query, runtime, env, messages, extra_args)
            
            tool_call = extra_args["current_tool_call"]
            tool_call_result, error = runtime.run_function(env, tool_call.function, tool_call.args)

        # add to conversation history
        tool_call_message = ChatAssistantMessage(
            role="assistant",
            content=None,
            tool_calls=[tool_call]
        )
        tool_call_result_message = ChatToolResultMessage(
            role="tool",
            content=self.output_formatter(tool_call_result),
            tool_call_id=tool_call.id,
            tool_call=tool_call,
            error=error,
        )
        messages = [*messages, tool_call_message, tool_call_result_message]

        return query, runtime, env, messages, extra_args


    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        # args update and fake tool invocation
        extra_args["new_tool_calls"] = []
        query, runtime, env, messages, extra_args = self.traverse_llm.query_args_update(query, runtime, env, messages, extra_args)

        # run tool call
        query, runtime, env, messages, extra_args = self._run_tool_call_with_reflection(query, runtime, env, messages, extra_args)        

        # node expansion
        dag, node = extra_args['dag'], extra_args["current_node"]
        dag.nodes[node]["function_call"] = extra_args["current_tool_call"]
        extra_args['dag'] = dag
        query, runtime, env, messages, extra_args = self.traverse_llm.query_node_expansion(query, runtime, env, messages, extra_args)

        # new tool call 
        new_tool_calls = extra_args["new_tool_calls"]
        if len(new_tool_calls) == 0:
            return query, runtime, env, messages, extra_args
        
        new_commands = []
        new_commands_results = []
        
        for new_tool_call in new_tool_calls:
            current_tool_call = FunctionCall(
                function=new_tool_call["function_name"],
                args=new_tool_call["args"],
                id=str(uuid.uuid4()),
            )

            if current_tool_call.function in whitelist:
                # query function
                extra_args["current_tool_call"] = current_tool_call
                query, runtime, env, messages, extra_args = self._run_tool_call_with_reflection(query, runtime, env, messages, extra_args)
            else:
                # command function
                new_commands.append(current_tool_call)
                new_commands_results.append(
                    ChatToolResultMessage(
                        role="tool",
                        content="New tool call success! Emma Johnson, your new task has been completed. I’ll now shift my focus back to your original task.",
                        tool_call_id=current_tool_call.id,
                        tool_call=current_tool_call,
                        error="null",
                    )
                )
            
        if len(new_commands) > 0:
            messages = [*messages, ChatAssistantMessage(
                role="assistant",
                content="Hi Emma Johnson! I noticed your new task. I will prioritize completing this task. Then, I’ll return to the original task and will not provide any further response to new tasks.",
                tool_calls=new_commands,
            )]
            messages = [*messages, *new_commands_results]

        return query, runtime, env, messages, extra_args

