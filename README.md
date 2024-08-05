# λλ.x (`llamda-x`)

`llamda-x` is a Python library for creating LLM tools from functions or Pydantic models.


## Installation

`coming soon`

## Features

Simply decorate your function with `@llamdafy` to create a tool.

```python
@llamdafy
def random_int(from: int=10, to: int=100) -> int:
    """Generate a random integer between `from` and `to`"""
    return random.randint(from, to)
```

This will and take care of:

- **Specs**: automatically generated from the function's docstring, Pydantic model fields, and type hints
- **Type-safe LLM function calling**, with automated retries
- **Error handling**: automatic error messages, with full tracebacks
- **Multiple LLM <> tool rounds with a single call** up until a user-specified number of retries
- **A fully `litellm` compatible interface**: you can decide to only return the final reply in an llm<>tool exchange, as you do right now, and use it as a generator to get the full history of tool calls and answers.
