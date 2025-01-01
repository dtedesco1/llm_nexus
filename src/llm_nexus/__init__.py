"""
This module provides a unified interface for multiple language model providers.
"""

__version__ = "0.2.0"  # Major version bump for breaking change: Anthropic API switch

# Import utility functions and classes
from ._utils.completions import CompletionParameters
from ._utils.function_calls import (
    Argument,
    Function,
    FunctionCallParameters,
    ParamType,
)

# Import main components
from .model_provider import ModelProvider

# Import provider classes
from .providers._anthropic_provider import AnthropicProvider
from .providers._googleai_provider import GoogleAIProvider
from .providers._openai_provider import OpenAIProvider

# You can also define __all__ to control what gets imported with "from llm_nexus import *"
__all__ = [
    "ModelProvider",
    "CompletionParameters",
    "Argument",
    "Function",
    "FunctionCallParameters",
    "ParamType",
    "AnthropicProvider",
    "GoogleAIProvider",
    "OpenAIProvider",
    "__version__",
]
