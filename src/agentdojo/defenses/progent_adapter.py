import importlib
import os
import sys
from collections.abc import Mapping, Sequence
from types import MethodType, ModuleType

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionCallArgTypes, FunctionsRuntime, TaskEnvironment
from agentdojo.types import ChatMessage

_SUITE_ALWAYS_ALLOWED_TOOLS = {
    "workspace": [
        "get_unread_emails",
        "get_sent_emails",
        "get_received_emails",
        "get_draft_emails",
        "search_emails",
        "search_contacts_by_name",
        "search_contacts_by_email",
        "get_current_day",
        "search_calendar_events",
        "get_day_calendar_events",
        "search_files_by_filename",
        "get_file_by_id",
        "list_files",
        "search_files",
    ],
    "travel": [
        "get_user_information",
        "get_all_hotels_in_city",
        "get_hotels_prices",
        "get_rating_reviews_for_hotels",
        "get_hotels_address",
        "get_all_restaurants_in_city",
        "get_cuisine_type_for_restaurants",
        "get_restaurants_address",
        "get_rating_reviews_for_restaurants",
        "get_dietary_restrictions_for_all_restaurants",
        "get_contact_information_for_restaurants",
        "get_price_for_restaurants",
        "check_restaurant_opening_hours",
        "get_all_car_rental_companies_in_city",
        "get_car_types_available",
        "get_rating_reviews_for_car_rental",
        "get_car_fuel_options",
        "get_car_rental_address",
        "get_car_price_per_day",
        "search_calendar_events",
        "get_day_calendar_events",
        "get_flight_information",
    ],
    "slack": [
        "get_channels",
        "read_channel_messages",
        "read_inbox",
        "get_users_in_channel",
    ],
}


def _load_secagent() -> ModuleType:
    """Load the untouched upstream package without its optional LangChain 1.x integration."""
    module_name = "agentdojo.defenses.progent.secagent"
    if module_name in sys.modules:
        return sys.modules[module_name]

    # Upstream only checks whether `langchain` imports, then unconditionally imports a
    # LangChain 1.x-only module. AgentDojo uses LangChain 0.x and does not use that
    # optional Progent integration, so hide it only while the upstream module loads.
    missing = object()
    existing_langchain = sys.modules.get("langchain", missing)
    sys.modules["langchain"] = None  # type: ignore[assignment]
    try:
        return importlib.import_module(module_name)
    finally:
        if existing_langchain is missing:
            sys.modules.pop("langchain", None)
        else:
            sys.modules["langchain"] = existing_langchain  # type: ignore[assignment]


class ProgentPolicyBootstrap(BasePipelineElement):
    """Connect the upstream Progent policy engine to the current AgentDojo runtime."""

    def __init__(self, suite_name: str | None) -> None:
        self.suite_name = suite_name
        self._tools_initialized = False
        self._wrapped_runtime_ids: set[int] = set()

    def _initialize_tools(self, runtime: FunctionsRuntime) -> ModuleType:
        secagent = _load_secagent()
        tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "args": tool.parameters.model_json_schema().get("properties", {}),
            }
            for tool in runtime.functions.values()
        ]

        secagent.reset_security_policy(include_human_policy=True)
        secagent.update_available_tools(tools)
        if self.suite_name == "banking":
            secagent.update_always_allowed_tools(
                ["get_most_recent_transactions"],
                allow_all_no_arg_tools=True,
            )
        elif self.suite_name in _SUITE_ALWAYS_ALLOWED_TOOLS:
            secagent.update_always_allowed_tools(_SUITE_ALWAYS_ALLOWED_TOOLS[self.suite_name])

        self._tools_initialized = True
        return secagent

    def _guard_runtime(self, runtime: FunctionsRuntime, secagent: ModuleType) -> None:
        runtime_id = id(runtime)
        if runtime_id in self._wrapped_runtime_ids:
            return

        original_run_function = runtime.run_function

        def guarded_run_function(
            self_runtime: FunctionsRuntime,
            call_env: TaskEnvironment | None,
            function_name: str,
            args: Mapping[str, FunctionCallArgTypes],
            raise_on_error: bool = False,
        ):
            try:
                secagent.check_tool_call(function_name, args)
            except Exception as error:
                if raise_on_error:
                    raise
                return "", f"{type(error).__name__}: {error}"

            result, error = original_run_function(
                call_env,
                function_name,
                args,
                raise_on_error=raise_on_error,
            )
            if os.getenv("SECAGENT_UPDATE", "False").lower() == "true" and error is None:
                secagent.generate_update_security_policy(
                    [{"name": function_name, "args": dict(args)}],
                    str([str(result)]),
                    manual_check=False,
                )
            return result, error

        runtime.run_function = MethodType(guarded_run_function, runtime)
        self._wrapped_runtime_ids.add(runtime_id)

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        secagent = self._initialize_tools(runtime) if not self._tools_initialized else _load_secagent()
        secagent.generate_security_policy(query)
        self._guard_runtime(runtime, secagent)
        return query, runtime, env, messages, extra_args
