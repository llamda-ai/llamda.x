"""Typer for communicating with the openai api"""

from typing import Literal
from litellm.types.utils import ChatCompletionMessageToolCall as OaiToolCall
from openai.types.chat import (
    ChatCompletionMessageParam as OaiMessage,
    ChatCompletionAssistantMessageParam as OaiAssistantMessage,
    ChatCompletionUserMessageParam as OaiUserMessage,
    ChatCompletionSystemMessageParam as OaiSystemMessage,
    ChatCompletionToolMessageParam as OaiToolMessage,
    ChatCompletionToolParam as OaiToolSpec,
    ChatCompletionMessageToolCallParam as OaiToolResult,
    ChatCompletion as OaiCompletion,
    ChatCompletionMessage as OaiResponseMessage,
)

from openai import OpenAI as OaiClient


__all__: list[str] = [
    "OaiCompletion",
    "OaiSystemMessage",
    "OaiToolCall",
    "OaiToolMessage",
    "OaiToolSpec",
    "OaiAssistantMessage",
    "OaiUserMessage",
    "OaiToolResult",
    "OaiResponseMessage",
    "OaiMessage",
    "OaiRole",
    "OaiRoleMessageMap",
    "OaiException",
    "OaiClient",
]

type OaiRole = Literal["user"] | Literal["system"] | Literal["assistant"] | Literal[
    "tool"
] | Literal["function"]


class OaiException(BaseException):
    """An exception type for LLM API Responses."""


OaiRoleMessageMap: dict[
    OaiRole, type[OaiUserMessage] | type[OaiSystemMessage] | type[OaiAssistantMessage]
] = {
    "user": OaiUserMessage,
    "system": OaiSystemMessage,
    "assistant": OaiAssistantMessage,
}
