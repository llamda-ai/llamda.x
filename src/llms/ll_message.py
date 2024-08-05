"""Classes related to Messages"""

import uuid
from typing import Any, Self, Sequence
from pydantic import BaseModel, Field
from rich.console import Console

from llamda.llms.ll_tool import LLToolCall, LLToolResponse
from .oai_api_types import (
    OaiRole,
    OaiCompletion,
    OaiMessage,
    OaiResponseMessage,
    OaiUserMessage,
    OaiSystemMessage,
    OaiAssistantMessage,
    OaiToolMessage,
)


__all__: list[str] = ["LLMessageMeta", "LLMessage"]

console = Console()


class LLMessageMeta(BaseModel):
    """Metadata for messages"""

    choice: dict[str, Any] | None = Field(exclude=True)
    completion: dict[str, Any] | None = Field(exclude=True)


class LLMessage(BaseModel):
    """Represents a message in an Exchange"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: OaiRole = "user"
    content: str = ""
    name: str | None = None
    tool_calls: Sequence[LLToolCall] | None = None
    meta: LLMessageMeta | None = None
    tool_call_id: str | None = None

    @classmethod
    def _message(cls, **kwargs: Any) -> OaiMessage:
        """
        Creates an OpenAI-compatible message from the provided keyword arguments.

        Args:
            **kwargs: Keyword arguments to create the LLMessage.

        Returns:
            OaiMessage: An OpenAI-compatible message object.
        """
        return oai_props(**kwargs)

    @property
    def oai_props(self) -> OaiMessage:
        """Returns the OpenAI-compatible message"""
        return oai_props(self)

    @classmethod
    def from_tool_response(cls, response: LLToolResponse) -> Self:
        """A message containing the result"""
        return LLMessage(
            id=response.tool_call_id,
            tool_call_id=response.tool_call_id,
            role="tool",
            content=response.result,
        )

    @classmethod
    def from_completion(cls, completion: OaiCompletion) -> Self:
        """Creates a Message from the first choice of an OpenAI-type completion request"""

        choice = completion.choices[0]
        message: OaiResponseMessage = choice.message

        tool_calls: list[LLToolCall] = (
            [LLToolCall.from_call(tc) for tc in message.tool_calls]
            if message.tool_calls
            else []
        )

        return cls(
            id=completion.id,
            meta=LLMessageMeta(
                choice=choice.model_dump(exclude={"message", "tool_call"}),
                completion=completion.model_dump(exclude={"choices"}),
            ),
            role=message.role,
            content=message.content or "",
            tool_calls=tool_calls,
        )


def oai_props(message: "LLMessage", **kwargs: Any) -> OaiMessage:
    """Rerutns the OpenAI API verison of a message"""
    if message.name:
        kwargs["name"] = message.name
    if message.content:
        kwargs["content"] = message.content
    match message.role:
        case "user":
            return OaiUserMessage(
                role="user",
                **kwargs,
            )
        case "system":
            return OaiSystemMessage(
                role="system",
                **kwargs,
            )
        case "assistant":
            if message.tool_calls:
                kwargs["tool_calls"] = [tc.oai for tc in message.tool_calls]
            return OaiAssistantMessage(
                role="assistant",
                **kwargs,
            )
        case "tool":
            if not message.id:
                raise ValueError("tool_call_id is required for tool messages")
            return OaiToolMessage(
                tool_call_id=message.id,
                role="tool",
                **kwargs,
            )
        case _:
            raise ValueError(f"Invalid role: {message.role}")
