import json
import re
from typing import Any, Dict, List

from anthropic import AI_PROMPT, HUMAN_PROMPT, Anthropic
from pydantic import BaseModel

from .._logging.log import log
from .._utils.completions import CompletionParameters
from .._utils.function_calls import Argument, Function, FunctionCallParameters
from ._model_interface import ModelInterface


class AnthropicProvider(ModelInterface, BaseModel):
    """
    Provides access to Anthropic's language models.

    Attributes:
        default_model (str): The default language model to use.
        default_temperature (float): The default temperature to use.
        default_max_tokens (int): The default maximum number of tokens to generate.
        api_key (str): The API key for the language model provider.

    Methods:
        completion(instructions: str, prompt: str, model: str, temperature: float, max_tokens: int) -> str:
            Generates a generic completion using the language model provider.
        function_call(user_prompt: str, function: Function, model: str, temperature: float) -> Union[str, Dict]:
            Generates a function call using the language model provider.
        function_result_check(function: Function, actual_result: Dict) -> bool:
            Checks if the actual result of a function call matches the function's expected arguments.
    """

    def __init__(self, api_key: str):
        Anthropic(api_key=api_key)

    def provider_class_completion(
        self,
        completion_parameters: CompletionParameters,
    ) -> str:
        """
        Generates a generic completion using the Anthropic language model provider.

        Args:
            completion_parameters (CompletionParameters): The parameters for the text completion.

        Returns:
            str: The generated completion.
        """

        prompt = f"{HUMAN_PROMPT} {completion_parameters.instructions}\n\nNow, please respond to the following prompt:\n{completion_parameters.prompt}{AI_PROMPT}"
        completion = Anthropic().completions.create(
            model=completion_parameters.model,
            max_tokens_to_sample=completion_parameters.max_tokens,
            prompt=prompt,
            temperature=completion_parameters.temperature,
        )
        return completion.completion

    # TODO: Update with enforceable function calling, based on May 17 update from Anthropic. https://docs.anthropic.com/en/docs/tool-use?=5-16_tool-use and https://docs.anthropic.com/en/docs/tool-use?=5-16_tool-use#forcing-tool-use
    def provider_class_function_call(
        self,
        function_call_parameters: FunctionCallParameters,
        return_array: bool = False,  # TODO: Add array function calling support
    ) -> List[Dict]:
        prompt = f"{HUMAN_PROMPT} You respond only with *valid* json.\n\nPlease respond to the following prompt:  {function_call_parameters.user_prompt}\n\n{function_call_parameters.function.to_dict()}\n\n You reply with a json object whose keys are *exactly* {function_call_parameters.function.expected_arguments()}, with values of each key in their appropriate types. {AI_PROMPT}"
        completion = Anthropic().completions.create(
            model=function_call_parameters.model,
            prompt=prompt,
            max_tokens_to_sample=10000,
            temperature=function_call_parameters.temperature,
        )

        # TODO Find a better way to do this, so that users can specify the max tokens.
        log.warning("Max tokens set to 10000 in _anthropic_provider.py")

        try:
            if isinstance(completion.completion, dict):
                result: Dict = completion.completion
            else:
                # Convert the response to a string
                response_str = str(completion.completion)
                # Use a regex to replace single quotes surrounding keys with double quotes
                response_str = re.sub(
                    r"(\s*{\s*|\s*,\s*)'([^']+)'\s*:", r'\1"\2":', response_str
                )
                result: Dict = json.loads(response_str)
        except json.decoder.JSONDecodeError as exception:
            log.error(
                "{}\nFunction call result did not return valid JSON. Please try again. What was returned:\n{}\n".format(
                    exception, completion.completion
                )
            )
            raise exception
        if not return_array:
            result = {"array": [result]}
        return result

    def provider_class_function_result_check(
        self, function: Function, actual_result: Dict
    ) -> bool:
        """
        Checks if the actual result of a function call matches the function's expected arguments.

        Args:
            function (Function): The function to check.
            actual_result (Dict): The actual result of the function call.

        Returns:
            bool: True if the actual result matches the function's expected arguments, False otherwise.
        """
        for arg in function.arguments:
            if arg.name not in actual_result["array"][0]:
                return False
        return True
