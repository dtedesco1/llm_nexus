# LLM Nexus

LLM Nexus is a Python package that provides a unified interface for interacting with multiple Language Model providers, including OpenAI, Anthropic, and Google AI.

## Features

- Unified API for multiple LLM providers
- Support for text completions and function calls
- Easy integration with OpenAI, Anthropic, and Google AI models

## Installation

You can install LLM Nexus using pip:

```
pip install llm-nexus
```

## Usage

Here's a quick example of how to use LLM Nexus:

```python
from llm_nexus import ModelProvider

provider = ModelProvider(openai_api_key="your-openai-key", anthropic_api_key="your-anthropic-key", googleai_api_key="your-googleai-key")

result = provider.completion(
    instructions="Return a rhyme.",
    user_prompt="Would a rose by any other name smell as sweet?",
    provider_name="openai",
    model_name="gpt-3.5-turbo"
)

print(result)
# Example response:  "Yes, a rose's fragrance can't be beat."
```

For function calls:

```python
from llm_nexus import ModelProvider, Function, Argument, ParamType

provider = ModelProvider(openai_api_key="your-openai-key", anthropic_api_key="your-anthropic-key", googleai_api_key="your-googleai-key")

function = Function(
    name="underrated_historical_events",
    description="Name up to 3 underrated historical events based on the user's request",
    arguments=[
        Argument(name="event", param_type=ParamType.STRING, description="The name of the event."),
        Argument(name="year", param_type=ParamType.INTEGER, description="The year the event began."),
        Argument(name="reasoning", param_type=ParamType.STRING, description="A brief description of why you feel this event is underrated.")
    ],
)

result = provider.function_call(
    user_prompt="What are some underrated A.D. events in African history?",
    function=function,
    provider_name="openai",
    model_name="gpt-3.5-turbo",
    return_array=True
)

print(result)
# Example reponse: 
# {
#     "array": [
#         {
#             "event": "Kingdom of Aksum",
#             "year": 100,
#             "reasoning": "One of the most powerful early medieval empires in East Africa, yet not widely known or studied in history classes."
#         },
#         {
#             "event": "Mali Empire",
#             "year": 1235,
#             "reasoning": "A prosperous West African empire centered in Mali that controlled trade routes across the Sahara. Advanced cities and universities but its achievements are not always emphasized."
#         },
#         {
#             "event": "Great Zimbabwe",
#             "year": 1450,
#             "reasoning": "Sophisticated city and trade center indicating a highly organized society, but origins are still debated and its influence on the region could be discussed more."
#         }
#     ]
# }
```

## Supported Python Versions

LLM Nexus supports Python 3.7 and above.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.