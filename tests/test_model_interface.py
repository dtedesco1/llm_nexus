import unittest
from typing import Dict

from pydantic import BaseModel

from src.llm_nexus._utils.completions import CompletionParameters
from src.llm_nexus._utils.function_calls import (
    Argument,
    Function,
    FunctionCallParameters,
    ParamType,
)
from src.llm_nexus.providers._model_interface import ModelInterface


class _ConcreteModel(ModelInterface, BaseModel):
    def __init__(self, api_key: str):
        pass

    def provider_class_completion(
        self,
        completion_parameters: CompletionParameters,
    ) -> str:
        return "Test completion"

    def provider_class_function_call(
        self,
        function_call_parameters: FunctionCallParameters,
    ) -> Dict:
        return {"Test": "function call"}

    def provider_class_function_result_check(
        self,
        function: Function,
        actual_result: Dict,
    ) -> bool:
        return True


class TestModelInterface(unittest.TestCase):
    def setUp(self):
        self.api_key = "your-api-key"
        self.model = _ConcreteModel(api_key=self.api_key)
        self.completion_parameters = CompletionParameters(
            instructions="Test instructions",
            prompt="Test prompt",
            model="Test model",
        )
        self.function_call_parameters = FunctionCallParameters(
            user_prompt="Test user prompt",
            function=Function(
                name="Test function",
                description="Test description",
                arguments=[
                    Argument(
                        name="Test argument",
                        description="Test description",
                        param_type=ParamType.STRING,
                    )
                ],
            ),
            model="Test model",
        )

    def test_provider_class_completion(self):
        result = self.model.provider_class_completion(self.completion_parameters)
        self.assertEqual(result, "Test completion")

    def test_provider_class_function_call(self):
        result = self.model.provider_class_function_call(self.function_call_parameters)
        self.assertEqual(result, {"Test": "function call"})
        self.assertIsInstance(result, Dict)

    def test_provider_class_function_result_check(self):
        result = self.model.provider_class_function_result_check(
            self.function_call_parameters.function, {"Test": "function call"}
        )
        self.assertTrue(result)
        self.assertIsInstance(result, bool)


if __name__ == "__main__":
    unittest.main()
