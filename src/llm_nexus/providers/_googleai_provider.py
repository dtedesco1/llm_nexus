import ast
import json
import re
from typing import Dict, List

import google.generativeai as genai
from google.generativeai.types import HarmBlockThreshold, HarmCategory
from pydantic import BaseModel

from .._logging.log import log
from .._utils.completions import CompletionParameters
from .._utils.function_calls import Argument, Function, FunctionCallParameters
from ._model_interface import ModelInterface

# TODO: Reconsider this if we start ingesting UGC or other unvetted content.
# SAFETY_SETTINGS = {
#     HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
#     HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
#     HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
#     HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
#     HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
# }
SAFETY_SETTINGS = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]


def convert_array_values(data: dict):
    """
    Converts string values in an array to their actual types.

    Args:
        data (Dict): The data to convert.

    Returns:
        Dict: The converted data.
    """
    for item in data["array"]:
        for key in item:
            try:
                item[key] = ast.literal_eval(item[key])
            except (ValueError, SyntaxError, TypeError):
                pass
    return data


class GoogleAIProvider(ModelInterface, BaseModel):
    """
    Provides access to Google's language models.

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
        genai.configure(api_key=api_key)

    def provider_class_completion(
        self,
        completion_parameters: CompletionParameters,
        candidate_count: int = 1,
        stop_sequences: List[str] = ["x"],
    ) -> str:
        """
        Generates a generic completion using a Google language model.

        Args:
            completion_parameters (CompletionParameters): The parameters for the text completion.
            candidate_count (int): The number of candidates to generate. Unique to Google AI.
            stop_sequences (List[str]): The list of stop sequences to use. Unique to Google AI.

        Returns:
            str: The generated completion.
        """
        model = genai.GenerativeModel(completion_parameters.model)
        response = model.generate_content(
            f"{completion_parameters.instructions}\n{completion_parameters.prompt}",
            generation_config=genai.types.GenerationConfig(
                # Only one candidate for now.
                candidate_count=candidate_count,
                stop_sequences=stop_sequences,
                max_output_tokens=completion_parameters.max_tokens,
                temperature=completion_parameters.temperature,
            ),
        )
        text = response.text

        if response.candidates[0].finish_reason.name == "MAX_TOKENS":
            text += "..."

        return text

    def provider_class_function_call(
        self,
        function_call_parameters: FunctionCallParameters,
        return_array: bool = False,
        candidate_count: int = 1,
        stop_sequences: List[str] = ["    "],
    ) -> Dict:
        """
        Generates a function call using a Google language model.

        Args:
            function_call_parameters (FunctionCallParameters): The parameters for the function call.
            return_array (bool): Whether to return multiple function calls or not.
            candidate_count (int): The number of candidates to generate. Unique to Google AI.
            stop_sequences (List[str]): The list of stop sequences to use. Unique to Google AI.

        Returns:
            Dict: The generated function call.
        """

        # arg_format = "{"
        # for arg in function_call_parameters.function.arguments[:-1]:
        #     arg_format += f"{arg.name}: {arg.description}, "
        # arg_format += f"{function_call_parameters.function.arguments[-1].name}: {function_call_parameters.function.arguments[-1].description}"
        # arg_format += "}"

        arg_format = function_call_parameters.function.arg_dict()

        if return_array is False:
            arg_format = {"array": [arg_format]}
            respond_as_if_text = (
                "Respond to provide inputs for a single call to the following function:"
            )
        else:
            arg_format = {"array": [arg_format, arg_format, arg_format, "..."]}

            respond_as_if_text = "Respond with an array of arguments representing multiple function calls to the following function:"

        function_call_prompt = f"""
            Respond *only* with pure, properly formatted JSON to the following prompt. Do not include any formatting. Prompt:
            
            {function_call_parameters.user_prompt}
            
            {respond_as_if_text}
            {function_call_parameters.function.to_dict()}

            Your response must follow this format:
            {json.dumps(arg_format)}

            Your JSON response:
            """

        model = genai.GenerativeModel(
            function_call_parameters.model,
            generation_config={"response_mime_type": "application/json"},
        )
        try:
            response = model.generate_content(
                f"{function_call_prompt}",
                generation_config=genai.types.GenerationConfig(
                    # Only one candidate for now.
                    candidate_count=candidate_count,
                    stop_sequences=stop_sequences,
                    max_output_tokens=function_call_parameters.max_tokens,
                    temperature=function_call_parameters.temperature,
                ),
                safety_settings=SAFETY_SETTINGS,
            )
            response_obj = response.text
        except Exception as e:
            log.error(
                f"Error generating response to function call prompt: {e}\n\nDetails:\n{e}\n\n"
            )
            raise ValueError(
                f"An error occurred while generating content. Please try again. Details:\n{e}\n"
            )

        if response.candidates[0].finish_reason.name == "MAX_TOKENS":
            raise ValueError(
                "The function call exceeded the max output token limit. Please try again."
            )

        try:
            if isinstance(response_obj, dict):
                result: Dict = response_obj
            else:
                # Convert the response to a string
                response_str = str(response_obj)
                # Use a regex to replace single quotes surrounding keys with double quotes
                response_str = re.sub(
                    r"(\s*{\s*|\s*,\s*)'([^']+)'\s*:", r'\1"\2":', response_str
                )
                result: Dict = json.loads(response_str)
        except json.JSONDecodeError:
            raise ValueError(
                f"The response was not properly formatted JSON. Please try again. Response:\n{response_obj}\n"
            )
        return result

    def provider_class_function_result_check(
        self, function: Function, function_calls: List[Dict]
    ) -> bool:
        """
        Checks if the actual result of a function call matches the function's expected arguments.

        Args:
            function (Function): The function to check.
            actual_result (Dict): The actual result of the function call.

        Returns:
            bool: True if the actual result matches the expected result, False otherwise.
        """
        try:
            # Check if the response has an array key
            array = function_calls["array"]
            for function_call in array:
                for argument in function.expected_arguments():
                    if argument not in function_call:
                        return False
        except KeyError:
            for argument in function.expected_arguments():
                if argument not in function_call:
                    return False
        return True
