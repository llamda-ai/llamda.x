"""
Defines the base classes for Llamda functions: LlamdaCallable and LlamdaBase.
These classes provide the foundation for creating and managing Llamda functions,
including abstract methods for execution and schema generation.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field
from ..llms.oai_api_types import OaiToolSpec


R = TypeVar("R")

__all__ = ["LlamdaCallable"]


class LlamdaCallable(Generic[R], ABC):
    """
    Represents a callable to proxy the original Function or model internally.
    This abstract base class defines the interface for Llamda functions.
    """

    @abstractmethod
    def run(self, **kwargs: Any) -> R:
        """
        Execute the Llamda function with the given parameters.
        This method should be implemented by subclasses.

        Args:
            **kwargs: Keyword arguments to be passed to the function.

        Returns:
            The result of the function execution.
        """
        pass

    @abstractmethod
    def to_tool_schema(self) -> OaiToolSpec:
        """
        Convert the Llamda function to a tool schema compatible with OpenAI's API.
        This method should be implemented by subclasses.

        Returns:
            A dictionary representing the OpenAI tool specification.
        """
        pass

    @classmethod
    @abstractmethod
    def create(
        cls,
        call_func: Callable[..., R],
        name: str = "",
        description: str = "",
        **kwargs: Any,
    ) -> "LlamdaCallable[R]":
        """
        Create a new LlamdaCallable instance.
        This method should be implemented by subclasses.

        Args:
            call_func: The function to be wrapped.
            name: The name of the Llamda function.
            description: A description of the Llamda function.
            **kwargs: Additional keyword arguments for function creation.

        Returns:
            A new instance of LlamdaCallable.
        """
        pass


class LlamdaBase(BaseModel, LlamdaCallable[R]):
    """
    The base class for Llamda functions, combining Pydantic's BaseModel
    with the LlamdaCallable interface.
    """

    name: str
    description: str
    call_func: Callable[..., R]
    fields: Dict[str, tuple[type, Any]] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def to_schema(self) -> Dict[str, Any]:
        """
        Get the JSON schema for the Llamda function.
        This method should be implemented by subclasses.

        Returns:
            A dictionary representing the JSON schema of the function.
        """
        raise NotImplementedError

    def to_tool_schema(self) -> OaiToolSpec:
        """Get the JSON schema for the LlamdaPydantic."""
        schema = self.to_schema()
        return {
            "type": "function",
            "function": {
                "name": schema["title"],
                "description": schema["description"],
                "parameters": {
                    "type": "object",
                    "properties": schema["properties"],
                    "required": schema.get("required", []),
                },
            },
        }

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        return self.to_schema()["properties"]
