from .. import llamdafy, litellm
from .prompts.schizopot import schizopot_sys_prompt
from .example_functions import aq_multiple

llamdafy(name="aq_multiple")(aq_multiple)


def run_example():
    messages = [{"content": schizopot_sys_prompt, "role": "system"}]
    while True:
        message: str = input("> ")
        messages.append({"content": message, "role": "user"})
        response = litellm.completion(
            "gpt-4o-mini", messages, tool_names=["aq_multiple"]
        )
        print(response)
