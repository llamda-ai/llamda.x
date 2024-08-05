# Spec

This is llamda-base, extracted from llamda-fn.

It is a library for simplifying work with function calling on LLMs. It provides:

1. A decorator to transform functions or Pydantic models into LLM tools - taking care of the json schema generation, the type inference etc.
2. A wrapper for the LiteLLM's completion API.
  The wrapper will accept a max_retries parameter, and will retry the request in case of a rate limit error.
   1. Accept a "tool_names" list, and pass only the tools in question to the completion API.
   2. Send the request and receive the API response and yield its content.
   3. if the response does not contain tool calls, also simply return the response as is.
   4. if the response contains tool calls:
      1. call the functions with the provided arguments.
      2. yield the result of the function call.
      3. go back to step 2.
   5. Return the result of the functin call to the API.