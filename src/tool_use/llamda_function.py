"""
Defines the LlamdaFunction class, which implements a Llamda function
using a simple function model as input. It extends the LlamdaBase class and
provides methods for creating, running, and generating schemas for function-based
Llamda functions with built-in validation using Pydantic models.
"""

from typing import Any, Callable, Dict, Type

from pydantic import BaseModel, Field, create_model

from .llamda_callable import LlamdaBase
from .process_fields import JsonDict
from .llamda_callable import R

__all__ = ["LlamdaFunction"]


class LlamdaFunction(LlamdaBase[R]):
    """
    A Llamda function that uses a simple function model as the input.
    This class provides a way to create Llamda functions with
    built-in validation using Pydantic models.
    """

    parameter_model: Type[BaseModel]

    @classmethod
    def create(
        cls,
        call_func: Callable[..., R],
        name: str = "",
        description: str = "",
        fields: JsonDict = {},
        **kwargs: Any,
    ) -> "LlamdaFunction[R]":
        """
        Create a new LlamdaFunction from a function.

        Args:
            call_func: The function to be called when running the Llamda function.
            name: The name of the Llamda function.
            description: A description of the Llamda function.
            fields: A dictionary of field names and their types/default values.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            A new LlamdaFunction instance.
        """
        model_fields = {}
        for field_name, (field_type, field_default) in fields.items():
            if field_default is ...:
                model_fields[field_name] = (field_type, Field(...))
            else:
                model_fields[field_name] = (field_type, Field(default=field_default))

        parameter_model: type[BaseModel] = create_model(
            f"{name}Parameters", **model_fields
        )

        return cls(
            name=name,
            description=description,
            parameter_model=parameter_model,
            call_func=call_func,
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            field_name: field_info
            for field_name, field_info in self.parameter_model.model_fields.items()
        }

    def run(self, **kwargs: Any) -> R:
        """
        Run the LlamdaFunction with the given parameters.

        Args:
            **kwargs: Keyword arguments to be validated and passed to the function.

        Returns:
            The result of the function execution.
        """
        validated_params = self.parameter_model(**kwargs)
        return self.call_func(**validated_params.model_dump())

    def to_schema(self) -> Dict[str, Any]:
        schema: dict[str, Any] = self.parameter_model.model_json_schema()
        schema["title"] = self.name
        schema["description"] = self.description
        return schema

    # @classmethod
    # def from_dict(
    #     cls, llamda_function_meta: Dict[str, Any], **kwargs: Any
    # ) -> "LlamdaFunction[R]":
    #     """
    #     Create a new LlamdaFunction from a dictionary.
    #     """
    #     return cls(
    #         call_func=load_function(llamda_function_meta["call_func"]),
    #         **llamda_function_meta,
    #     )

    def to_dict(self) -> Dict[str, Any]:
        """
        Get the dictionary representation of the LlamdaFunction.

        Returns:
            A dictionary representing the LlamdaFunction.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "call_func": {
                "module": self.call_func.__module__,
                "name": self.call_func.__name__,
            },
        }

    @property
    def __name__(self) -> str:
        return self.name
