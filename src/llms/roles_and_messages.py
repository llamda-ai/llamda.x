from typing import Literal

from .oai_api_types import OaiUserMessage, OaiSystemMessage, OaiAssistantMessage

type OaiRole = Literal["user"] | Literal["system"] | Literal["assistant"] | Literal[
    "tool"
] | Literal["function"]


OaiRoleMessageMap: dict[
    OaiRole, type[OaiUserMessage] | type[OaiSystemMessage] | type[OaiAssistantMessage]
] = {
    "user": OaiUserMessage,
    "system": OaiSystemMessage,
    "assistant": OaiAssistantMessage,
}
