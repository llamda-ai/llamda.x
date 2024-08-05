"""
Main entry point.

Exports:
- `llamdafy`: A function for converting a regular function into a Llamda function.
- `tool_completion`: A wrapper around the Litellm completion function.
- `litellm`: A copy of the Litellm client with the tools added.

If you prefer to only import the specific completion function,
you can import the unchanged `litellm` client with `import litellm`.
"""

import copy
import litellm
from .tool_use.llamda_tools import LlamdaTools
from .tool_use.llamdafy import llamdafy

from . import litellm_adapter


llamda_tools = LlamdaTools()

tool_completion = litellm_adapter.completion_wrapper(litellm.completion)

litellm_with_tools = copy.copy(litellm)
litellm_with_tools.completion = tool_completion
litellm = litellm_with_tools
__all__ = ["llamdafy", "tool_completion", "litellm"]
