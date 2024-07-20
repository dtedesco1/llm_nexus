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
```

For more detailed usage instructions, please refer to the documentation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
