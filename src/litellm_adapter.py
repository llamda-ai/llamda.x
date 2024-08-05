# Wraps the LitLLM completion function to allow for tool calls.

from functools import wraps
from typing import Any, Callable, Generator, TypeAlias, TypeVar, cast

import litellm
from git import Sequence
from litellm import Choices, CustomStreamWrapper, Message
from litellm.types.utils import ModelResponse
from .llms.ll_tool import LLToolResponse
from .llms.oai_api_types import OaiToolMessage
from .tool_use.llamda_callable import LlamdaBase
from .tool_use.llamda_tools import LlamdaTools

WrappedFn = TypeVar(
    "WrappedFn", bound=Callable[..., CustomStreamWrapper | ModelResponse]
)
ReturnType: TypeAlias = CustomStreamWrapper | ModelResponse
WrappedCompletionReturnType: TypeAlias = (
    Generator[Message, Any, ReturnType] | ReturnType
)


functions: dict[str, Callable[..., Any]] = {}
tool_names: list[str] = []


@wraps(litellm.completion)
def completion_wrapper(f: WrappedFn) -> WrappedFn:
    """
    Wraps the LitLLM completion function to allow for tool calls.
    """

    def completion(
        model: str,
        tool_names: list[str],
        messages: list[OaiToolMessage | Message],
        *args: Any,
        **kwargs: Any,
    ) -> WrappedCompletionReturnType:
        llamda_tools = LlamdaTools()

        active_tools: dict[str, LlamdaBase[Any]] = (
            {name: llamda_tools.tools[name] for name in tool_names}
            if len(tool_names) > 0
            else llamda_tools.tools
        )

        # not using tools, or no tools available: proxy.
        if (
            kwargs.get("stream")
            or kwargs.get("stream_options")
            or kwargs.get("tool_choice") == "none"
            or len(functions) == 0
            or len(active_tools) == 0
        ):
            return f(model=model, messages=messages, *args, **kwargs)

        # using tools: get the response.
        response: ModelResponse | CustomStreamWrapper = litellm.completion(
            *args, **kwargs
        )

        # if the response is a stream (it should not happen, but...) return it
        if isinstance(response, CustomStreamWrapper):
            return response

        choices: Sequence[Choices] = response.choices  # type: ignore

        message: Message = choices[0].message
        messages = [*messages, message]

        # if there are multiple choices, print a warning.
        if len(choices) > 1 and choices[0].message.tool_calls is not None:
            raise Warning(
                f"""Multiple choices in response: {choices}.\
                Only the first one will be used for tool calls."""
            )

        # yield the message, to be sure.
        yield message

        # no tool calls; return
        if message.tool_calls is None:
            return response
        else:
            for tool_call in message.tool_calls:
                result: LLToolResponse = llamda_tools.execute_function(
                    tool_call=tool_call
                )

                messages.append(result.oai)

            new_response: WrappedCompletionReturnType = completion(
                model=model, tool_names=tool_names, messages=messages, *args, **kwargs
            )
            return new_response

    return cast(WrappedFn, completion)


__all__ = ["completion_wrapper"]
