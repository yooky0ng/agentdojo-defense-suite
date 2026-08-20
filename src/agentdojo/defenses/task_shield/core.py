import json
import logging
from collections.abc import Mapping

import openai
from json_repair import loads as repair_json

from agentdojo.defenses.task_shield.prompts import (
    CONTENT_CHECKER_SYSTEM_PROMPT,
    CONTENT_CHECKER_USER_PROMPT,
    CONTENT_MISALIGNMENT_FEEDBACK,
    TASK_EXTRACTION_SYSTEM_PROMPT,
    TASK_EXTRACTION_USER_PROMPT,
    TOOL_CALL_CHECKER_SYSTEM_PROMPT,
    TOOL_CALL_CHECKER_USER_PROMPT,
    TOOL_OUTPUT_MISALIGNMENT_FEEDBACK,
    USER_INTENTIONS_REMINDER,
)

logger = logging.getLogger(__name__)


class TaskShieldModel:
    """OpenAI defender model used for extraction and alignment scoring."""

    def __init__(self, client: openai.OpenAI, model: str) -> None:
        self.client = client
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str, history: str = "") -> str:
        if history:
            user_prompt = f"Conversation History:\n{history}\n\n{user_prompt}"
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content or ""


def _json_array(text: str) -> list:
    try:
        value = repair_json(text, skip_json_loads=True)
    except Exception:
        logger.warning("Task Shield could not parse defender JSON", exc_info=True)
        return []
    return value if isinstance(value, list) else []


class TaskShield:
    """Minimal implementation of Algorithm 1 and the three checks in Section 4.2."""

    def __init__(self, model: TaskShieldModel) -> None:
        self.model = model
        self.user_tasks: list[str] = []

    def reset(self, user_query: str) -> None:
        self.user_tasks = self.extract(user_query, "the user")
        if not self.user_tasks and user_query.strip():
            self.user_tasks = [user_query.strip()]

    def extract(self, content: str, subject: str) -> list[str]:
        if not content.strip():
            return []
        answer = self.model.generate(
            TASK_EXTRACTION_SYSTEM_PROMPT,
            TASK_EXTRACTION_USER_PROMPT.format(subject=subject, message_content=content),
        )
        return [item.strip() for item in _json_array(answer) if isinstance(item, str) and item.strip()]

    def instruction_is_aligned(
        self,
        instruction: str,
        level: str,
        history: str,
        tool_information: str = "",
    ) -> bool:
        answer = self.model.generate(
            CONTENT_CHECKER_SYSTEM_PROMPT,
            CONTENT_CHECKER_USER_PROMPT.format(
                actionable_instruction=instruction,
                current_level=level,
                tool_information=tool_information,
                task_instructions_text=self._user_tasks_text(),
            ),
            history,
        )
        return self._score_sum(answer) > 0

    def tool_call_is_aligned(
        self,
        tool_name: str,
        tool_args: Mapping[str, object],
        related_content: str,
        history: str,
    ) -> bool:
        answer = self.model.generate(
            TOOL_CALL_CHECKER_SYSTEM_PROMPT,
            TOOL_CALL_CHECKER_USER_PROMPT.format(
                tool_name=tool_name,
                tool_args=json.dumps(dict(tool_args), ensure_ascii=False),
                related_content=related_content,
                task_instructions_text=self._user_tasks_text(),
            ),
            history,
        )
        return self._score_sum(answer) > 0

    def feedback(self, instructions: list[str], tool_call_id: str | None = None) -> str:
        blocks = [
            CONTENT_MISALIGNMENT_FEEDBACK.format(misaligned_instruction=instruction) for instruction in instructions
        ]
        if tool_call_id is not None:
            blocks.append(TOOL_OUTPUT_MISALIGNMENT_FEEDBACK.format(tool_call_id=tool_call_id))
        blocks.append(USER_INTENTIONS_REMINDER.format(user_instructions=self._user_tasks_text()))
        return "\n".join(blocks)

    def _score_sum(self, answer: str) -> float:
        scores = []
        for item in _json_array(answer):
            if not isinstance(item, Mapping):
                continue
            try:
                scores.append(float(item["ContributeTo score"]))
            except (KeyError, TypeError, ValueError):
                continue
        if not scores:
            logger.warning("Task Shield received no valid ContributeTo score; allowing the item")
            return 1.0
        return sum(scores)

    def _user_tasks_text(self) -> str:
        return "\n".join(f"- {task}" for task in self.user_tasks)
