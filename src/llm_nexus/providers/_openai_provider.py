"""
This module provides access to OpenAI's language models through the OpenAIProvider class.

The OpenAIProvider class provides methods for generating generic completions and function calls using the language model provider.
"""

import json
import logging
import re
from typing import Dict, List

import openai
from pydantic import BaseModel

from .._logging.log import log
from .._utils.completions import CompletionParameters
from .._utils.function_calls import Function, FunctionCallParameters

# from openai.error import InvalidRequestError
from ._model_interface import ModelInterface


class OpenAIProvider(ModelInterface, BaseModel):
    """
    Provides access to OpenAI's language models.

    Attributes:
        default_model (str): The default language model to use.
        default_temperature (float): The default temperature to use.
        default_max_tokens (int): The default maximum number of tokens to generate.
        api_key (str): The API key for the language model provider.

    Methods:
        completion(completion_parameters: CompletionParameters) -> str:
            Generates a generic completion using the language model provider.
        function_call(function_call_parameters: FunctionCallParameters) -> Dict:
            Generates a function call using the language model provider.
        function_result_check(function: Function, actual_result: Dict) -> bool:
            Checks if the actual result of a function call matches the function's expected arguments.
    """

    def __init__(self, api_key: str):
        openai.api_key = api_key

    def provider_class_completion(
        self,
        completion_parameters: CompletionParameters,
    ) -> str:
        """
        Generates a generic completion using the language model provider.

        Args:
            completion_parameters (CompletionParameters): The parameters for the text completion.

        Returns:
            str: The generated completion.
        """
        response = openai.chat.completions.create(
            model=completion_parameters.model,
            temperature=completion_parameters.temperature,
            max_tokens=completion_parameters.max_tokens,
            messages=[
                {"role": "system", "content": completion_parameters.instructions},
                {"role": "user", "content": completion_parameters.prompt},
            ],
        )

        content: str = response.choices[0].message.content

        return content

    def provider_class_function_call(
        self,
        function_call_parameters: FunctionCallParameters,
        return_array: bool = False,
    ) -> Dict:
        """
        Generates a function call using the language model provider.

        Args:
            function_call_parameters (FunctionCallParameters): The parameters for the function call.
            return_array (bool): Whether to return an array of function calls or just one.

        Returns:
            Dict: The generated function call.
        """
        response = openai.chat.completions.create(
            model=function_call_parameters.model,
            messages=[
                {
                    "role": "system",
                    "content": "*Strictly* follow the function guidelines and formats. You do not include extra arguments, or deviate from the arguments requested.",
                },
                {"role": "user", "content": function_call_parameters.user_prompt},
            ],
            tools=self._format_function_call(
                function_call_parameters.function, return_array
            ),
            tool_choice={
                "type": "function",
                "function": {"name": function_call_parameters.function.name},
            },
        )
        content = response.choices[0].message.tool_calls[0].function.arguments
        log.debug(f"OpenAI Function call response: {content}")
        try:
            if isinstance(content, dict):
                result: Dict = content
            else:
                # Convert the response to a string
                response_str = str(content)
                # Use a regex to replace single quotes surrounding keys with double quotes
                response_str = re.sub(
                    r"(\s*{\s*|\s*,\s*)'([^']+)'\s*:", r'\1"\2":', response_str
                )
                result: Dict = json.loads(response_str)
        # Print the results if the response is not valid JSON
        except json.decoder.JSONDecodeError as exception:
            logging.critical(
                f"Function call result did not return valid JSON. Please try again. What was returned:\n{content}\n\nException:\n{exception}"
            )
            return None
        if not return_array:
            result = {"array": [result]}
        return result

    def provider_class_function_result_check(
        self, function: Function, function_calls: List[Dict]
    ) -> bool:
        """
        Checks if each function call's arguments match a function's expected arguments.

        Args:
            function (Function): The function to check against.
            function_calls (List[Dict]): The list of function call arguments to check.

        Returns:
            bool: True if each function call's arguments match the expected arguments, False otherwise.
        """
        try:
            # Check if the response has an array key
            array = function_calls["array"]
            for function_call in array:
                for argument in function.expected_arguments():
                    if argument not in function_call:
                        logging.error(
                            f"Argument {argument} not found in function call {function_call}"
                        )
                        return False
        except (KeyError, TypeError):
            for argument in function.expected_arguments():
                if argument not in function_calls:
                    return False
        return True

    def _format_function_call(
        self, function: Function, return_array: bool
    ) -> List[Dict]:
        """
        Formats a function call request for the OpenAI API.

        Args:
            function (Function): The function to format.
            return_array (bool): Whether to return an array of function calls or not.

        Returns:
            List[Dict]: The formatted function call.
        """

        properties = {}
        for argument in function.arguments:
            properties[argument.name] = {
                "type": argument.param_type.value,
                "description": argument.description,
            }

        required = []
        for prop in properties:
            required.append(prop)

        if return_array:
            function_call = [
                {
                    "type": "function",
                    "function": {
                        "name": function.name,
                        "description": function.description,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "array": {  # Our name for the array
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": properties,
                                        "required": required,
                                    },
                                }
                            },
                            "required": ["array"],
                        },
                    },
                }
            ]

        if not return_array:
            function_call = [
                {
                    "type": "function",
                    "function": {
                        "name": function.name,
                        "description": function.description,
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required,
                        },
                    },
                }
            ]
        return function_call
