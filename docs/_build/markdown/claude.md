# File list

## llamda.py

Main entry point for the library.

## llamda_fn/llms

Files for working with LLMs.

### api_types.py

import uuid
from functools import cached_property
from typing import Any, Literal, Self, List

from openai.types.chat import ChatCompletion as OaiCompletion
from openai.types.chat import ChatCompletionToolParam as OaiToolParam
from openai.types.chat import ChatCompletionMessageParam as OaiRequestMessage
from openai.types.chat import ChatCompletionAssistantMessageParam as OaiAssistantMessage
from openai.types.chat import ChatCompletionUserMessageParam as OaiUserMessage
from openai.types.chat import ChatCompletionSystemMessageParam as OaiSystemMessage
from openai.types.chat import ChatCompletionMessageToolCall as OaiToolCall
from openai.types.chat import ChatCompletionFunctionCallOptionParam as OaiToolFunction
from pydantic import BaseModel, Field

Role = Literal[“user”, “system”, “assistant”, “tool”]

class LlToolCall(BaseModel):
: id: str
  name: str
  arguments: str
  <br/>
  @classmethod
  def from_oai_tool_call(cls, call: OaiToolCall) -> Self:
  <br/>
  > return cls(
  > : id=call.id,
  >   name=call.function.name,
  >   arguments=call.function.arguments,
  <br/>
  > )

class ToolResponse(BaseModel):
: id: str
  name: str
  arguments: str
  \_result: str
  <br/>
  def \_\_init_\_(self, result: str = “”, 
  <br/>
  ```
  **
  ```
  <br/>
  kwargs: Any) -> None:
  : super()._\_init_\_(
    <br/>
    ```
    **
    ```
    <br/>
    kwargs)
    self._result = result
  <br/>
  @cached_property
  def result(self) -> str:
  <br/>
  > if isinstance(self._result, BaseModel):
  > : return self._result.model_dump_json()
  <br/>
  > else:
  > : return self._result

def make_oai_role_message(
: role: Role,
  content: str,
  name: str | None = None,
  tool_calls: List[LlToolCall] | None = None,
  <br/>
  ```
  **
  ```
  <br/>
  kwargs: Any,

) -> OaiUserMessage | OaiSystemMessage | OaiAssistantMessage:
: kwargs = {}
  if name:
  <br/>
  > kwargs[“name”] = name
  <br/>
  match role:
  : case “user”:
    : return OaiUserMessage(
      : content=content,
        <br/>
        ```
        **
        ```
        <br/>
        kwargs,
      <br/>
      )
    <br/>
    case “system”:
    : return OaiSystemMessage(
      : content=content,
        <br/>
        ```
        **
        ```
        <br/>
        kwargs,
      <br/>
      )
    <br/>
    case “assistant”:
    <br/>
    > if tool_calls:
    > : kwargs[“tool_calls”] = [
    >   : tool_call.model_dump() for tool_call in tool_calls
    >   <br/>
    >   ]
    <br/>
    > return OaiAssistantMessage(
    > : content=content,
    >   <br/>
    >   ```
    >   **
    >   ```
    >   <br/>
    >   kwargs,
    <br/>
    > )
    <br/>
    case \_:
    : raise ValueError(f”Invalid role: {role}”)

OaiRoleMessage: dict[
: Role, type[OaiUserMessage] | type[OaiSystemMessage] | type[OaiAssistantMessage]

] = {
: “user”: OaiUserMessage,
  “system”: OaiSystemMessage,
  “assistant”: OaiAssistantMessage,

}

class LLMessageMeta(BaseModel):
: choice: dict[str, Any] | None = Field(exclude=True)
  completion: dict[str, Any] | None = Field(exclude=True)

class LLMessage(BaseModel):
: id: str = Field(default_factory=uuid.uuid4)
  role: Role
  content: str
  name: str | None = None
  tool_calls: List[LlToolCall] | None = None
  meta: LLMessageMeta | None = None
  <br/>
  def get_oai_message(self):
  : return make_oai_role_message(
    : self.role, self.content, self.name, self.tool_calls
    <br/>
    )
  <br/>
  @classmethod
  def from_execution(cls, execution: ToolResponse) -> Self:
  <br/>
  > return cls(
  > : role=”tool”,
  >   id=execution.id,
  >   name=execution.name,
  >   content=execution.result,
  <br/>
  > )

class LLCompletion(BaseModel):
: message: LLMessage
  meta: LLMessageMeta | None = None
  <br/>
  @classmethod
  def from_completion(cls, completion: OaiCompletion) -> Self:
  <br/>
  > choice = completion.choices[0]
  > message = choice.message
  > tool_calls = None
  > if message.tool_calls:
  <br/>
  > > tool_calls = [
  > > : LlToolCall.from_oai_tool_call(tc) for tc in message.tool_calls
  <br/>
  > > ]
  <br/>
  > return cls(
  > : message=LLMessage(
  >   : id=completion.id,
  >     meta=LLMessageMeta(
  >     <br/>
  >     > choice=choice.model_dump(exclude={“message”}),
  >     > completion=completion.model_dump(exclude={“choices”}),
  >     <br/>
  >     ),
  >     role=message.role,
  >     content=message.content or “”,
  >     tool_calls=tool_calls,
  >   <br/>
  >   )
  <br/>
  > )

class OaiRequest(BaseModel):
: messages: list[OaiRequestMessage]
  tools: list[OaiToolParam]

\_\_all_\_ = [
: “LLMessage”,
  “LLCompletion”,
  “OaiCompletion”,
  “OaiToolParam”,
  “OaiToolFunction”,
  “OaiToolCall”,

]

### llm_manager.py

from typing import Any
from pydantic import Field, model_validator
from openai import OpenAI
from openai.types.chat import ChatCompletion
from .api_types import LLCompletion, LLMessage
from .type_transformers import make_oai_message
from .api import LlmApiConfig

class LLManager(OpenAI):
: api_config: dict[str, Any] = Field(default_factory=dict)
  llm_name: str = Field(default=”gpt-4-0613”)
  <br/>
  def \_\_init_\_(
  : self,
    llm_name: str = “gpt-4-0613”,
    <br/>
    ```
    **
    ```
    <br/>
    kwargs: Any,
  <br/>
  ):
  : self.llm_name = llm_name
    super()._\_init_\_(
    <br/>
    ```
    **
    ```
    <br/>
    kwargs)
  <br/>
  class Config:
  : arbitrary_types_allowed = True
  <br/>
  def chat_completion(self, messages: list[LLMessage], 
  <br/>
  ```
  **
  ```
  <br/>
  kwargs: Any) -> LLCompletion:
  <br/>
  > oai_completion: ChatCompletion = super().chat.completions.create(
  > : messages=[make_oai_message(
  >   <br/>
  >   ```
  >   **
  >   ```
  >   <br/>
  >   msg.model_dump()) for msg in messages],
  >   model=self.llm_name,
  <br/>
  > )
  > return LLCompletion.from_completion(oai_completion)
  <br/>
  @model_validator(mode=”before”)
  @classmethod
  def validate_api_and_model(cls, data: dict[str, Any]) -> dict[str, Any]:
  <br/>
  > “””Validate the API and model.”””
  > api_config = data.get(“api_config”) or {}
  > api = (
  <br/>
  > > data.get(“api”)
  > > if isinstance(data.get(“api”), OpenAI)
  > > else LlmApiConfig(
  <br/>
  > > ```
  > > **
  > > ```
  <br/>
  > > api_config).create_openai_client()
  <br/>
  > )
  > if not api or not isinstance(api, OpenAI):
  <br/>
  > > raise ValueError(“Unable to create OpenAI client.”)
  <br/>
  > data.update({“api”: api})
  <br/>
  > if data.get(“llm_name”):
  > : available_models: list[str] = [model.id for model in api.models.list()]
  >   if data.get(“llm_name”) not in available_models:
  >   <br/>
  >   > raise ValueError(
  >   > : f”Model ‘{data.get(‘llm_name’)}’ is not available. ”
  >   >   f”Available models: {’, ‘.join(available_models)}”
  >   <br/>
  >   > )
  <br/>
  > else:
  > : raise ValueError(“No LLM API client or LLM name provided.”)
  <br/>
  > return data

### type_transformers.py

from typing import Any, List
from llamda_fn.llms.api_types import (

> Role,
> LlToolCall,
> OaiUserMessage,
> OaiSystemMessage,
> OaiAssistantMessage,

)

def make_oai_message(
: role: Role,
  content: str,
  name: str | None = None,
  tool_calls: List[LlToolCall] | None = None,
  <br/>
  ```
  **
  ```
  <br/>
  kwargs: Any,

) -> OaiUserMessage | OaiSystemMessage | OaiAssistantMessage:
: kwargs = {}
  if name:
  <br/>
  > kwargs[“name”] = name
  <br/>
  match role:
  : case “user”:
    : return OaiUserMessage(
      : content=content,
        <br/>
        ```
        **
        ```
        <br/>
        kwargs,
      <br/>
      )
    <br/>
    case “system”:
    : return OaiSystemMessage(
      : content=content,
        <br/>
        ```
        **
        ```
        <br/>
        kwargs,
      <br/>
      )
    <br/>
    case “assistant”:
    <br/>
    > if tool_calls:
    > : kwargs[“tool_calls”] = [
    >   : tool_call.model_dump() for tool_call in tool_calls
    >   <br/>
    >   ]
    <br/>
    > return OaiAssistantMessage(
    > : content=content,
    >   <br/>
    >   ```
    >   **
    >   ```
    >   <br/>
    >   kwargs,
    <br/>
    > )
    <br/>
    case \_:
    : raise ValueError(f”Invalid role: {role}”)

OaiRoleMessage: dict[
: Role, type[OaiUserMessage] | type[OaiSystemMessage] | type[OaiAssistantMessage]

] = {
: “user”: OaiUserMessage,
  “system”: OaiSystemMessage,
  “assistant”: OaiAssistantMessage,

}

### exchange.py

from llamda_fn.llms.api_types import LLMessage

from collections import UserList
from typing import List, Optional

class Exchange(UserList[LLMessage]):
: “””
  An exchange represents a series of messages between a user and an assistant.
  “””
  <br/>
  def \_\_init_\_(
  : self,
    system_message: Optional[str] = None,
    messages: Optional[List[LLMessage]] = None,
  <br/>
  ) -> None:
  : “””
    Initialize the exchange.
    “””
    super()._\_init_\_()
    if system_message:
    <br/>
    > self.data.append(LLMessage(content=system_message, role=”system”))
    <br/>
    if messages:
    : self.data.extend(messages)
  <br/>
  def ask(self, content: str) -> None:
  : “””
    Add a user message to the exchange.
    “””
    self.data.append(LLMessage(content=content, role=”user”))
  <br/>
  def append(self, item: LLMessage) -> None:
  : “””
    Add a message to the exchange.
    “””
    <br/>
    self.data.append(item)
  <br/>
  def get_context(self, n: int = 5) -> list[LLMessage]:
  : “””
    Get the last n messages as context.
    “””
    return self.data[-n:]
  <br/>
  def \_\_str_\_(self) -> str:
  : “””
    String representation of the exchange.
    “””
    return “n”.join(f”{msg.role}: {msg.content}” for msg in self.data)

## llamda_fn/functions

### function_types.py

from typing import Any, Callable, Dict, Generic, TypeVar, Type
from pydantic import BaseModel, Field, create_model, ConfigDict

from llamda_fn.llms import OaiToolParam

R = TypeVar(“R”)

class LlamdaCallable(Generic[R]):
: def run(self, 
  <br/>
  ```
  **
  ```
  <br/>
  kwargs: Any) -> R:
  : raise NotImplementedError
  <br/>
  def to_tool_schema(self) -> OaiToolParam:
  : raise NotImplementedError
  <br/>
  @classmethod
  def create(
  <br/>
  > cls,
  > call_func: Callable[…, R],
  > fields: Dict[str, tuple[type, Any]],
  > name: str = “”,
  > description: str = “”,
  <br/>
  > ```
  > **
  > ```
  <br/>
  > kwargs: Any,
  <br/>
  ) -> “LlamdaCallable[R]”:
  : raise NotImplementedError

class LlamdaBase(BaseModel, LlamdaCallable[R]):
: “””The base class for Llamda functions.”””
  <br/>
  name: str
  description: str
  call_func: Callable[…, R]
  <br/>
  model_config = ConfigDict(arbitrary_types_allowed=True)
  <br/>
  def to_schema(self) -> Dict[str, Any]:
  : “””Get the JSON schema for the Llamda function.”””
    raise NotImplementedError
  <br/>
  def to_tool_schema(self) -> OaiToolParam:
  : “””Get the JSON schema for the LlamdaPydantic.”””
    schema = self.to_schema()
    return {
    <br/>
    > “type”: “function”,
    > “function”: {
    <br/>
    > > “name”: schema[“title”],
    > > “description”: schema[“description”],
    > > “parameters”: {
    <br/>
    > > > “type”: “object”,
    > > > “properties”: schema[“properties”],
    > > > “required”: schema.get(“required”, []),
    <br/>
    > > },
    <br/>
    > },
    <br/>
    }

class LlamdaFunction(LlamdaBase[R]):
: “””A Llamda function that uses a simple function model as the input.”””
  <br/>
  parameter_model: Type[BaseModel]
  <br/>
  @classmethod
  def create(
  <br/>
  > cls,
  > call_func: Callable[…, R],
  > fields: Dict[str, tuple[type, Any]],
  > name: str = “”,
  > description: str = “”,
  <br/>
  > ```
  > **
  > ```
  <br/>
  > kwargs: Any,
  <br/>
  ) -> “LlamdaFunction[R]”:
  : “””Create a new LlamdaFunction from a function.”””
    model_fields = {}
    for field_name, (field_type, field_default) in fields.items():
    <br/>
    > if field_default is …:
    > : model_fields[field_name] = (field_type, Field(…))
    <br/>
    > else:
    > : model_fields[field_name] = (field_type, Field(default=field_default))
    <br/>
    parameter_model = create_model(f”{name}Parameters”, 
    <br/>
    ```
    **
    ```
    <br/>
    model_fields)
    <br/>
    return cls(
    : name=name,
      description=description,
      parameter_model=parameter_model,
      call_func=call_func,
    <br/>
    )
  <br/>
  def run(self, 
  <br/>
  ```
  **
  ```
  <br/>
  kwargs: Any) -> R:
  : “””Run the LlamdaFunction with the given parameters.”””
    validated_params = self.parameter_model(
    <br/>
    ```
    **
    ```
    <br/>
    kwargs)
    return self.call_func(
    <br/>
    ```
    **
    ```
    <br/>
    validated_params.model_dump())
  <br/>
  def to_schema(self) -> Dict[str, Any]:
  : “””Get the JSON schema for the LlamdaFunction.”””
    schema = self.parameter_model.model_json_schema()
    schema[“title”] = self.name
    schema[“description”] = self.description
    return schema

class LlamdaPydantic(LlamdaBase[R]):
: “””A Llamda function that uses a Pydantic model as the input.”””
  <br/>
  model: Type[BaseModel]
  <br/>
  @classmethod
  def create(
  <br/>
  > cls,
  > call_func: Callable[…, R],
  > fields: Dict[str, tuple[type, Any]],
  > name: str = “”,
  > description: str = “”,
  <br/>
  > ```
  > **
  > ```
  <br/>
  > kwargs: Any,
  <br/>
  ) -> “LlamdaPydantic[R]”:
  : “””Create a new LlamdaPydantic from a Pydantic model.”””
    return cls(
    <br/>
    > name=name,
    > description=description,
    > model=model,
    > call_func=func,
    <br/>
    )
  <br/>
  def run(self, 
  <br/>
  ```
  **
  ```
  <br/>
  kwargs: Any) -> R:
  : “””Run the LlamdaPydantic with the given parameters.”””
    validated_params = self.model(
    <br/>
    ```
    **
    ```
    <br/>
    kwargs)
    return self.call_func(validated_params)
  <br/>
  def to_schema(self) -> dict[str, Any]:
  : “””Get the JSON schema for the LlamdaPydantic.”””
    schema = self.model.model_json_schema()
    schema[“title”] = self.name
    schema[“description”] = self.description
    return schema

### process_fields.py

from ast import List
from typing import Any, Dict, Union, get_args, get_origin
from pydantic import BaseModel, Field, ValidationError, create_model
from pydantic.fields import FieldInfo
from pydantic_core import SchemaError

JsonDict = Dict[str, Any]

def process_field(
: field_type: Any, field_info: Union[JsonDict, FieldInfo]

) -> tuple[Any, JsonDict]:
: “””
  Process a field type and info, using Pydantic’s model_json_schema for schema generation.
  “””
  try:
  <br/>
  > if isinstance(field_type, type) and issubclass(field_type, BaseModel):
  > : # Handle nested Pydantic models
  >   nested_schema = field_type.model_json_schema()
  >   field_schema = {
  >   <br/>
  >   > “type”: “object”,
  >   > “properties”: nested_schema.get(“properties”, {}),
  >   > “required”: nested_schema.get(“required”, []),
  >   <br/>
  >   }
  <br/>
  > else:
  > : # Create a temporary model with the field
  >   if isinstance(field_info, FieldInfo):
  >   <br/>
  >   > temp_field = field_info
  >   <br/>
  >   else:
  >   : temp_field = Field(
  >     <br/>
  >     ```
  >     **
  >     ```
  >     <br/>
  >     field_info)
  >   <br/>
  >   TempModel = create_model(“TempModel”, field=(field_type, temp_field))
  >   <br/>
  >   # Get the JSON schema for the entire model
  >   full_schema = TempModel.model_json_schema()
  >   <br/>
  >   # Extract the schema for our specific field
  >   field_schema = full_schema[“properties”][“field”]
  <br/>
  > # Handle Optional types
  > origin = get_origin(field_type)
  > if origin is Union:
  <br/>
  > > args = get_args(field_type)
  > > if type(None) in args:
  <br/>
  > > > # This is an Optional type
  > > > non_none_type = next(arg for arg in args if arg is not type(None))
  > > > if non_none_type is float:
  <br/>
  > > > > field_schema = {“type”: “number”, “nullable”: True}
  <br/>
  > > > elif non_none_type is int:
  > > > : field_schema = {“type”: “integer”, “nullable”: True}
  <br/>
  > > > elif non_none_type is str:
  > > > : field_schema = {“type”: “string”, “nullable”: True}
  <br/>
  > > > elif isinstance(non_none_type, type) and issubclass(
  > > > : non_none_type, BaseModel
  <br/>
  > > > ):
  > > > : field_schema = {“type”: “object”, “nullable”: True}
  <br/>
  > # Ensure ‘type’ is always set
  > if “type” not in field_schema:
  <br/>
  > > if isinstance(field_type, type) and issubclass(field_type, BaseModel):
  > > : field_schema[“type”] = “object”
  <br/>
  > > elif field_type is int:
  > > : field_schema[“type”] = “integer”
  <br/>
  > > elif field_type is float:
  > > : field_schema[“type”] = “number”
  <br/>
  > > elif field_type is str:
  > > : field_schema[“type”] = “string”
  <br/>
  > > elif field_type is bool:
  > > : field_schema[“type”] = “boolean”
  <br/>
  > > elif field_type is list or field_type is List:
  > > : field_schema[“type”] = “array”
  <br/>
  > > elif field_type is dict or field_type is Dict:
  > > : field_schema[“type”] = “object”
  <br/>
  > > else:
  > > : field_schema[“type”] = “any”
  <br/>
  > # Remove ‘title’ field if present
  > field_schema.pop(“title”, None)
  <br/>
  > # Merge field_info with the generated schema
  > if isinstance(field_info, dict):
  <br/>
  > > for key, value in field_info.items():
  > > : if key not in field_schema or field_schema[key] is None:
  > >   : field_schema[key] = value
  <br/>
  > return field_type, field_schema
  <br/>
  except (SchemaError, ValidationError) as e:
  : print(f”Error processing field: {e}”)
    return Any, {“type”: “any”, “error”: str(e)}

def process_fields(fields: Dict[str, Any]) -> Dict[str, tuple[Any, JsonDict]]:
: “””
  Process all fields in a model, using Pydantic for complex types.
  “””
  processed_fields = {}
  for field_name, field_value in fields.items():
  <br/>
  > if isinstance(field_value, FieldInfo):
  > : field_type = field_value.annotation
  >   field_info = field_value
  <br/>
  > elif isinstance(field_value, tuple):
  > : field_type, field_info = field_value
  <br/>
  > else:
  > : raise ValueError(
  >   : f”Unexpected field value type for {field_name}: {type(field_value)}”
  >   <br/>
  >   )
  <br/>
  > processed_type, processed_info = process_field(field_type, field_info)
  <br/>
  > # Ensure ‘type’ is set for nested Pydantic models
  > if isinstance(processed_type, type) and issubclass(processed_type, BaseModel):
  <br/>
  > > processed_info[“type”] = “object”
  <br/>
  > processed_fields[field_name] = (processed_type, processed_info)
  <br/>
  return processed_fields

### process_functions.py

import json
from inspect import Parameter, isclass, signature
from typing import (

> Any,
> Callable,
> Dict,
> List,
> Optional,
> TypeVar,
> ParamSpec,
> Sequence,
> Iterator,

)

from pydantic import BaseModel, ValidationError
from llamda_fn.llms.api_types import LlToolCall, ToolResponse, OaiToolParam
from .llamda_classes import LlamdaFunction, LlamdaPydantic, LlamdaCallable

R = TypeVar(“R”)
P = ParamSpec(“P”)

class LlamdaFunctions:
: def \_\_init_\_(self) -> None:
  : self._tools: Dict[str, LlamdaCallable[Any]] = {}
  <br/>
  @property
  def tools(self) -> Dict[str, LlamdaCallable[Any]]:
  <br/>
  > return self._tools
  <br/>
  def llamdafy(
  : self,
    name: Optional[str] = None,
    description: Optional[str] = None,
  <br/>
  ) -> Callable[[Callable[P, R]], LlamdaCallable[R]]:
  : def decorator(func: Callable[P, R]) -> LlamdaCallable[R]:
    : func_name: str = name or func._\_name_\_
      func_description: str = description or func._\_doc_\_ or “”
      <br/>
      sig = signature(func)
      if len(sig.parameters) == 1:
      <br/>
      > param = next(iter(sig.parameters.values()))
      > if isclass(param.annotation) and issubclass(
      <br/>
      > > param.annotation, BaseModel
      <br/>
      > ):
      > : llamda_func: LlamdaCallable[R] = LlamdaPydantic.create(
      >   : func_name, param.annotation, func_description, func
      >   <br/>
      >   )
      >   self._tools[func_name] = llamda_func
      >   return llamda_func
      <br/>
      fields: Dict[str, tuple[type, Any]] = {
      : param_name: (
        : param.annotation if param.annotation != Parameter.empty else Any,
          param.default if param.default != Parameter.empty else …,
        <br/>
        )
        for param_name, param in sig.parameters.items()
      <br/>
      }
      <br/>
      llamda_func: LlamdaCallable[R] = LlamdaFunction.create(
      : call_func=func,
        fields=fields,
        name=func_name,
        description=func_description,
      <br/>
      )
      self._tools[func_name] = llamda_func
      return llamda_func
    <br/>
    return decorator
  <br/>
  def get(self, names: Optional[List[str]] = None) -> Sequence[OaiToolParam]:
  : “””Returns the tool spec for some or all of the functions in the registry”””
    if names is None:
    <br/>
    > names = list(self._tools.keys())
    <br/>
    return [
    : self._tools[name].to_tool_schema() for name in names if name in self._tools
    <br/>
    ]
  <br/>
  def execute_function(self, tool_call: LlToolCall) -> ToolResponse:
  : “””Executes the function specified in the tool call with the required arguments”””
    try:
    <br/>
    > if tool_call.name not in self._tools:
    > : raise KeyError(f”Function ‘{tool_call.name}’ not found”)
    <br/>
    > parsed_args = json.loads(tool_call.arguments)
    > result = self._tools[tool_call.name].run(
    <br/>
    > ```
    > **
    > ```
    <br/>
    > parsed_args)
    <br/>
    except KeyError as e:
    : result = {“error”: f”Error: {str(e)}”}
    <br/>
    except ValidationError as e:
    : result = {“error”: f”Error: Validation failed - {str(e)}”}
    <br/>
    except Exception as e:
    : result = {“error”: f”Error: {str(e)}”}
    <br/>
    return ToolResponse(
    : id=tool_call.id,
      name=tool_call.name,
      arguments=tool_call.arguments,
      result=json.dumps(result),
    <br/>
    )
  <br/>
  def \_\_getitem_\_(self, key: str) -> LlamdaCallable[Any]:
  : return self._tools[key]
  <br/>
  def \_\_contains_\_(self, key: str) -> bool:
  : return key in self._tools
  <br/>
  def \_\_len_\_(self) -> int:
  : return len(self._tools)
  <br/>
  def \_\_iter_\_(self) -> Iterator[str]:
  : return iter(self._tools)
