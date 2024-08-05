"""
Defines the LlamdaTools class, which serves as a registry and manager
for Llamda functions. It provides functionality to register, execute, and manage
Llamda functions, including conversion of regular functions to Llamda functions
and generation of OpenAI-compatible tool specifications.
"""

import json
from inspect import Parameter, Signature, isclass, signature
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    ParamSpec,
    Sequence,
    TypeVar,
)

from pydantic import BaseModel, ValidationError

from ..llms.ll_tool import LLToolResponse
from ..llms.oai_api_types import OaiToolCall, OaiToolSpec

from .llamda_callable import LlamdaBase, LlamdaCallable
from .llamda_function import LlamdaFunction
from .llamda_pydantic import LlamdaPydantic

R = TypeVar("R")
P = ParamSpec("P")

__all__ = ["LlamdaTools"]


class LlamdaTools:
    """
    A registry and manager for Llamda functions.
    This class provides methods to register, execute, and manage Llamda functions.
    """

    _instance = None
    _tools: Dict[str, LlamdaBase[Any]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools = {}
        return cls._instance

    def __init__(self):
        # The __init__ method can be empty or contain any initialization code
        # that should only run once
        pass

    @property
    def tools(self) -> Dict[str, LlamdaBase[Any]]:
        """
        Returns a dictionary of all registered Llamda functions.
        """
        return self._tools

    def llamdafy(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Callable[[Callable[P, R]], LlamdaBase[R]]:
        """
        A decorator to convert a regular function into a Llamda function.

        This method analyzes the function signature and creates either a LlamdaPydantic
        or LlamdaFunction instance based on the input parameters.

        Args:
            name: Optional custom name for the Llamda function.
            description: Optional description for the Llamda function.

        Returns:
            A decorator that converts the function into a LlamdaCallable.
        """

        def decorator(func: Callable[P, R]) -> LlamdaBase[R]:
            func_name: str = name or func.__name__
            func_description: str = description or func.__doc__ or ""

            sig: Signature = signature(func)
            if len(sig.parameters) == 1:
                param: Parameter = next(iter(sig.parameters.values()))
                if isclass(param.annotation) and issubclass(
                    param.annotation, BaseModel
                ):
                    llamda_func = LlamdaPydantic.create(
                        call_func=func,
                        name=func_name,
                        description=func_description,
                        model=param.annotation,
                    )
                    self._tools[func_name] = llamda_func
                    return llamda_func

            fields: Dict[str, tuple[type, Any]] = {
                param_name: (
                    param.annotation if param.annotation != Parameter.empty else Any,
                    param.default if param.default != Parameter.empty else ...,
                )
                for param_name, param in sig.parameters.items()
            }

            llamda_func: LlamdaCallable[R] = LlamdaFunction.create(
                call_func=func,
                fields=fields,
                name=func_name,
                description=func_description,
            )
            self._tools[func_name] = llamda_func
            return llamda_func

        return decorator

    @property
    def spec(self) -> Sequence[OaiToolSpec]:
        """
        Returns the tool spec for all of the functions in the registry.

        Returns:
            A sequence of OaiToolSpec objects representing the functions.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": func.name,
                    "description": func.description,
                    "parameters": func.to_schema(),
                },
            }
            for func in self._tools.values()
        ]

    def get_spec(self, names: Optional[List[str]] = None) -> Sequence[OaiToolSpec]:
        """
        Returns the tool spec for some or all of the functions in the registry.

        Args:
            names: Optional list of function names to include in the spec.

        Returns:
            A sequence of OaiToolSpec objects representing the specified functions.
        """
        if names is None:
            return self.spec
        return [
            self._tools[name].to_tool_schema() for name in names if name in self._tools
        ]

    def execute_function(self, tool_call: OaiToolCall) -> LLToolResponse:
        """
        Executes the function specified in the tool call with the required arguments.

        This method handles various exceptions that might occur during execution
        and returns appropriate error messages.

        Args:
            tool_call: An LLToolCall object containing the function name and arguments.

        Returns:
            An LLToolResponse object containing the execution result or error information.
        """
        try:
            if tool_call.function.name not in self._tools:
                raise KeyError(f"Function '{tool_call.function.name}' not found")

            parsed_args = json.loads(tool_call.function.arguments)
            result = self._tools[tool_call.function.name].run(**parsed_args)
            success = True
        except KeyError as e:
            result = {"error": f"Error: {str(e)}"}
            success = False
        except ValidationError as e:
            result = {"error": f"Error: Validation failed - {str(e)}"}
            success = False
        except Exception as e:
            result = {"error": f"Error: {str(e)}"}
            success = False

        return LLToolResponse(
            id=tool_call.id,
            result=json.dumps(result),
            success=success,
        )

    def __getitem__(self, key: str) -> LlamdaCallable[Any]:
        return self._tools[key]

    def __contains__(self, key: str) -> bool:
        return key in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self) -> Iterator[str]:
        return iter(self._tools)
