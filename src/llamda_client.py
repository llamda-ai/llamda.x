# 1. Initialise a LlamdaFunctions as a singleton
# 2. Import litellm_adapter
# 3. Wrap the completion function with the completion_wrapper function
# 4. Export: the LlamdaFunctions singleton, the wrapped completion function,
#    and a copy of litellm with completion replaced with the wrapped function
