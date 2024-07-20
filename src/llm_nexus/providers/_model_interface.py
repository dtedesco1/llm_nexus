from abc import ABC, abstractmethod
from typing import Dict, List

from pydantic import BaseModel

from .._utils.completions import CompletionParameters
from .._utils.function_calls import Function, FunctionCallParameters


class ModelInterface(ABC, BaseModel):
    """
    This is the interface that all model providers must implement.

    The interface is abstract and must be implemented by a concrete class.

    It is designed to be used by the ModelProvider class, which is the class that the user will interact with.

    Attributes:
        api_key (str): The API key for the model provider.

    Methods:
        provider_class_completion(completion_parameters: CompletionParameters) -> str:
            This method is responsible for making a text completion call to the model provider's API and returning the result.
        provider_class_function_call(function_call_parameters: FunctionCallParameters) -> Dict:
            This method is responsible for making a function call to the model provider's API and returning the result.
        provider_class_function_result_check(function: Function, actual_result: Dict) -> bool:
            Checks if the actual result of a function call matches the function's expected arguments.
    """

    @abstractmethod
    def __init__(self, api_key: str):
        pass

    @abstractmethod
    def provider_class_completion(
        self,
        completion_parameters: CompletionParameters,
    ) -> str:
        """
        This method is responsible for making a text completion call to the model provider's API and returning the result.

        Args:
            completion_parameters (CompletionParameters): The parameters for the text completion.

        Returns:
            str: The result of the text completion call.
        """

    @abstractmethod
    def provider_class_function_call(
        self,
        function_call_parameters: FunctionCallParameters,
        return_array: bool,
    ) -> List[Dict]:
        """
        This method is responsible calling the model provider's API to generate arguments to one or more function calls and returning the results.

        Args:
            function_call_parameters (FunctionCallParameters): The parameters for the function call.
            return_array (bool): Whether to return multiple function calls or not.

        Returns:
            List[Dict]: A list of dictionaries, each containing arguments for a function call.
        """

    @abstractmethod
    def provider_class_function_result_check(
        self, function: Function, actual_result: List[Dict]
    ) -> bool:
        """
        Checks if the actual result of a function call matches the function's expected arguments.

        Args:
            function (Function): The function to check.
            actual_result (Dict): The actual result of the function call.

        Returns:
            bool: True if the actual result matches the expected result, False otherwise.
        """
