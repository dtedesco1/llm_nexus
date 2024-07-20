"""
This module provides the ModelProvider class for LLM Nexus.

The ModelProvider class offers a unified interface for interacting with
multiple Language Model providers, including OpenAI, Anthropic, and Google AI.
It supports text completions and function calls across different providers.
"""

import os
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from ._logging.log import log
from ._utils.completions import CompletionParameters
from ._utils.function_calls import Function, FunctionCallParameters
from .providers._anthropic_provider import AnthropicProvider
from .providers._googleai_provider import GoogleAIProvider
from .providers._model_interface import ModelInterface
from .providers._openai_provider import OpenAIProvider


class ModelProvider(BaseModel):
    """
    Provides access to different language models.

    Attributes:
        default_provider (str): The default language model provider to use.
        default_model (str): The default language model to use.
        default_temperature (float): The default temperature to use.
        default_max_tokens (int): The default maximum number of tokens to generate.
        providers (Dict[str, _ModelInterface]): The language model providers.

    Methods:
        completion(instructions: str, prompt: str, provider: str, model: str, temperature: float, max_tokens: int) -> str:
            Generates a generic completion using the language model provider.
        function_call(user_prompt: str, function: Function, provider: str, model: str, temperature: float) -> Dict:
            Generates a function call using the language model provider.
    """

    # TODO: Set up a CallSettings class to handle defaults
    default_provider: str = "openai"
    default_model: str = "gpt-3.5-turbo"
    default_temperature: float = 0.8
    default_max_tokens: int = 1000
    providers: Dict[str, ModelInterface] = {}

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        googleai_api_key: Optional[str] = None,
        **data: Any,
    ):
        super().__init__(**data)

        # Try to set up the providers with environment variables
        openai_api_key = os.getenv("OPENAI_API_KEY")
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        googleai_api_key = os.getenv("GOOGLEAI_API_KEY")

        if openai_api_key:
            self.providers["openai"] = OpenAIProvider(openai_api_key)

        if anthropic_api_key:
            self.providers["anthropic"] = AnthropicProvider(anthropic_api_key)

        if googleai_api_key:
            self.providers["googleai"] = GoogleAIProvider(googleai_api_key)

    def _check_provider_setup(self, provider: str) -> bool:
        if provider in self.providers:
            return True
        return False

    def completion(
        self,
        instructions: str,
        user_prompt: str,
        provider_name: str,
        model_name: str,
        temperature: float = default_temperature,
        max_tokens: int = default_max_tokens,
    ) -> str:
        """
        Generates a generic completion using the language model provider.

        Args:
            instructions (str): The instructions for the chat completion.
            user_prompt (str): The prompt for the chat completion.
            provider_name (str): The language model provider to use.
            model_name (str): The language model to use.
            temperature (float): The temperature to use for the chat completion.
            max_tokens (int): The maximum number of tokens to generate.

        Returns:
            str: The generated completion.
        """

        provider: ModelInterface = self.providers[provider_name]
        return provider.provider_class_completion(
            completion_parameters=CompletionParameters(
                instructions=instructions,
                prompt=user_prompt,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )

    def function_call(
        self,
        user_prompt: str,
        function: Function,
        provider_name: str,
        model_name: str,
        return_array: bool = False,
        temperature: float = default_temperature,
        max_tokens: int = default_max_tokens,
    ) -> Dict[str, List[Dict]]:
        """
        Generates a function call using the language model provider. Checks if the actual result of a function call matches the function's expected arguments. Tries up to 3 times to get a valid result, then raises an error if the result does not match the expected arguments.

        Args:
            user_prompt (str): The prompt for the function call.
            function (Function): The function to call.
            return_array (bool): Whether to return an array of function calls or just one.
            provider (str): The language model provider to use.
            model (str): The language model to use.
            temperature (float): The temperature to use for the function call.

        Returns:
            Dict[str, List[Dict]]: The generated function call result. The top level key is 'array'. Each element in the array is a dictionary of the function call result.
        """
        provider: ModelInterface = self.providers[provider_name]

        # Create function call parameters
        function_call_parameters = FunctionCallParameters(
            user_prompt=user_prompt,
            function=function,
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Try up to 3 times to get a valid result
        check = False
        i = 0
        while i <= 3 and not check:
            try:
                result: Dict[str, List[Dict]] = provider.provider_class_function_call(
                    function_call_parameters=function_call_parameters,
                    return_array=return_array,
                )
                log.debug(
                    "%s Model function call result received for function %s. %d elements in the result.",
                    model_name,
                    function.name,
                    len(result),
                )
                check = provider.provider_class_function_result_check(function, result)
                i += 1
            except (ValueError, RuntimeError) as e:
                log.warning(
                    "Error generating function call for %s:\n```%s```\nTrying again...",
                    model_name,
                    e,
                )
                i += 1
        if not check:
            log.error(
                "Function call result for %s does not match expected arguments.\nExpected: %s\nActual: %s",
                model_name,
                function.arguments,
                result,
            )
            raise ValueError(
                f"Function call result for {model_name} does not match expected arguments.\nExpected: {function.arguments}\nActual: {result}"
            )

        return result
