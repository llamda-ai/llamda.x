"""
This module contains the `llamdafy` function,
which is used to convert a regular function or Pydantic model
into a Llamda function.
"""

from typing import Any, Callable, Dict, Optional, TypeVar, ParamSpec
from inspect import Parameter, Signature, isclass, signature

from pydantic import BaseModel

from .llamda_tools import LlamdaTools

from .llamda_callable import LlamdaBase, LlamdaCallable
from .llamda_pydantic import LlamdaPydantic
from .llamda_function import LlamdaFunction


R = TypeVar("R")
P = ParamSpec("P")

__all__ = ["llamdafy"]


def llamdafy(
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
    tools = LlamdaTools()

    def decorator(func: Callable[P, R]) -> LlamdaBase[R]:
        func_name: str = name or func.__name__
        func_description: str = description or func.__doc__ or ""

        sig: Signature = signature(func)
        if len(sig.parameters) == 1:
            param: Parameter = next(iter(sig.parameters.values()))
            if isclass(param.annotation) and issubclass(param.annotation, BaseModel):
                llamda_func = LlamdaPydantic.create(
                    call_func=func,
                    name=func_name,
                    description=func_description,
                    model=param.annotation,
                )
                tools.add_tool(llamda_func)
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
        tools.add_tool(llamda_func)
        return llamda_func

    return decorator
