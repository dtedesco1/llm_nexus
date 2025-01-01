import json
import re
from typing import Any, Dict, List

from anthropic import Anthropic
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

        prompt = f"{completion_parameters.instructions}\n\nNow, please respond to the following prompt:\n{completion_parameters.prompt}"
        completion = Anthropic().messages.create(
            model=completion_parameters.model,
            max_tokens=completion_parameters.max_tokens,
            messages=[{"role": "user", "content": prompt}],
            temperature=completion_parameters.temperature,
        )
        return completion.content[0].text

    # TODO: Update with enforceable function calling, based on May 17 update from Anthropic. https://docs.anthropic.com/en/docs/tool-use?=5-16_tool-use and https://docs.anthropic.com/en/docs/tool-use?=5-16_tool-use#forcing-tool-use
    def provider_class_function_call(
        self,
        function_call_parameters: FunctionCallParameters,
        return_array: bool = False,
    ) -> List[Dict]:
        prompt = f"You must respond with a complete, valid JSON object. Your response should contain *only* the JSON, with no additional text.\n\nPlease respond to the following prompt: {function_call_parameters.user_prompt}\n\n{function_call_parameters.function.to_dict()}\n\n You must reply with a JSON object whose keys are *exactly* {function_call_parameters.function.expected_arguments()}, with values of each key in their appropriate types."

        if return_array:
            prompt = f"You must respond with a complete, valid JSON array containing exactly 3 items. Your response should contain *only* the JSON array, with no additional text.\n\nPlease respond to the following prompt: {function_call_parameters.user_prompt}\n\n{function_call_parameters.function.to_dict()}\n\n You must reply with a JSON array of objects whose keys are *exactly* {function_call_parameters.function.expected_arguments()}, with values of each key in their appropriate types."

        # Use a reasonable max_tokens value that's within Claude-3's limits
        max_tokens = min(function_call_parameters.max_tokens or 8192, 8192)

        completion = Anthropic().messages.create(
            model=function_call_parameters.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=function_call_parameters.temperature,
        )

        try:
            response_str = completion.content[0].text.strip()
            if isinstance(response_str, dict):
                result: Dict = response_str
            else:
                # Use a regex to replace single quotes surrounding keys with double quotes
                response_str = re.sub(
                    r"(\s*{\s*|\s*,\s*)'([^']+)'\s*:", r'\1"\2":', response_str
                )
                
                # Try to parse the JSON
                try:
                    result: Dict = json.loads(response_str)
                except json.decoder.JSONDecodeError:
                    # If the JSON is invalid, try to fix common issues
                    # 1. Remove any text before the first { or [ and after the last } or ]
                    response_str = re.sub(r'^[^{\[]*', '', response_str)
                    response_str = re.sub(r'[^}\]]*$', '', response_str)
                    
                    # 2. Try to parse again
                    result: Dict = json.loads(response_str)

        except json.decoder.JSONDecodeError as exception:
            log.error(
                "%s\nFunction call result did not return valid JSON. Please try again. What was returned:\n%s\n",
                exception,
                response_str,
            )
            # Attempt to self-heal the response with a more explicit prompt
            log.info("Attempting to self-heal the response.")
            completion_parameters = CompletionParameters(
                instructions=f"Fix this broken JSON to match this schema exactly: {function_call_parameters.function.expected_arguments()}\nYou must output ONLY the fixed JSON, with no additional text. Ensure all strings are properly quoted and terminated.",
                prompt=f"Fix this broken JSON:\n{response_str}",
                model=function_call_parameters.model,
                temperature=0.0,  # Use temperature 0 for more deterministic JSON fixing
            )
            self_heal_attempt = self.provider_class_completion(completion_parameters)
            log.info("Self-heal attempt result: %s\n", self_heal_attempt)
            
            # Try to parse the self-healed response
            try:
                if isinstance(self_heal_attempt, dict):
                    result: Dict = self_heal_attempt
                else:
                    # Clean up the self-healed response and ensure it's a complete array
                    self_heal_attempt = re.sub(r'^[^{\[]*', '', self_heal_attempt)
                    self_heal_attempt = re.sub(r'[^}\]]*$', '', self_heal_attempt)
                    if not self_heal_attempt.endswith(']'):
                        self_heal_attempt += ']'
                    result: Dict = json.loads(self_heal_attempt)
            except (json.decoder.JSONDecodeError, ValueError) as e:
                raise ValueError(
                    f"Anthropic function call result did not return valid JSON, even after self-healing attempt. Original response:\n{response_str}\n\nSelf-heal attempt:\n{self_heal_attempt}\n"
                ) from e

        if not return_array:
            result = {"array": [result] if not isinstance(result, list) else result}
        elif not isinstance(result, list):
            result = [result]

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
