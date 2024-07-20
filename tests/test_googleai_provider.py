import os
import unittest
from unittest.mock import MagicMock, patch

from src.llm_nexus._logging.log import log
from src.llm_nexus._utils.completions import CompletionParameters
from src.llm_nexus._utils.function_calls import (
    Argument,
    Function,
    FunctionCallParameters,
    ParamType,
)
from src.llm_nexus.providers._googleai_provider import GoogleAIProvider


class MockFinishReason:
    def __init__(self, name):
        self.name = name


class MockCandidate:
    def __init__(self, finish_reason):
        self.finish_reason = finish_reason


class MockResponse:
    def __init__(self, text, candidates):
        self.text = text
        self.candidates = candidates


test_model = "gemini-1.5-pro-latest"


class TestGoogleAIProvider(unittest.TestCase):
    def setUp(self):
        self.api_key = os.environ.get("GOOGLEAI_API_KEY")
        self.provider = GoogleAIProvider(api_key=self.api_key)

        # Create Argument objects
        arg1 = Argument(
            name="Year", param_type="integer", description="The year of the event."
        )
        arg2 = Argument(
            name="Event", param_type="string", description="The name of the event."
        )

        # Create Function object with lists of Arguments
        self.function = Function(
            name="UnderratedHistoricalEvent",
            description="Returns an underrated historical event.",
            arguments=[arg1, arg2],
        )

        # Create FunctionCallParameters and Function objects with lists of Arguments
        self.function_call_parameters = FunctionCallParameters(
            user_prompt="What is an underrated historical event?",
            function=self.function,
            model=test_model,
            temperature=0.8,
            max_tokens=2000,
        )
        # Write a call to return an array of historical events.
        self.function_call_parameters_array = FunctionCallParameters(
            user_prompt="What are three underrated historical events?",
            function=self.function,
            return_array=True,
            model="gemini-1.5-pro-latest",
            temperature=0.8,
            max_tokens=2000,
        )

    def test_provider_class_completion(self):
        completion_parameters = MagicMock()
        completion_parameters.model = "test-model"
        completion_parameters.prompt = "test-prompt"
        completion_parameters.max_tokens = 100
        completion_parameters.temperature = 0.8

        mock_model = MagicMock()
        mock_model.generate_content.return_value = MagicMock(text="test-generated-text")

        with patch(
            "llm_nexus.providers._googleai_provider.genai.GenerativeModel",
            return_value=mock_model,
        ):
            result = self.provider.provider_class_completion(completion_parameters)

        self.assertEqual(result, "test-generated-text")

    def test_provider_class_completion_with_custom_stop_sequences(self):
        completion_parameters = MagicMock()
        completion_parameters.model = "test-model"
        completion_parameters.prompt = "test-prompt"
        completion_parameters.max_tokens = 100
        completion_parameters.temperature = 0.8

        mock_model = MagicMock()
        mock_model.generate_content.return_value = MagicMock(
            text="test-generated-text...",
            candidates=[MagicMock(finish_reason=MagicMock(name="MAX_TOKENS"))],
        )

        with patch(
            "llm_nexus.providers._googleai_provider.genai.GenerativeModel",
            return_value=mock_model,
        ):
            result = self.provider.provider_class_completion(
                completion_parameters, stop_sequences=["x", "y"]
            )

        self.assertEqual(result, "test-generated-text...")

    def test_provider_class_completion_with_multiple_candidates(self):
        completion_parameters = MagicMock()
        completion_parameters.model = "test-model"
        completion_parameters.prompt = "test-prompt"
        completion_parameters.max_tokens = 100
        completion_parameters.temperature = 0.8

        mock_model = MagicMock()
        mock_model.generate_content.return_value = MagicMock(
            text="test-generated-text...",
            candidates=[
                MagicMock(finish_reason=MagicMock(name="MAX_TOKENS")),
                MagicMock(finish_reason=MagicMock(name="MAX_TOKENS")),
            ],
        )

        with patch(
            "llm_nexus.providers._googleai_provider.genai.GenerativeModel",
            return_value=mock_model,
        ):
            result = self.provider.provider_class_completion(
                completion_parameters, candidate_count=2
            )

        self.assertEqual(result, "test-generated-text...")

    def test_real_completion_call(self):
        completion_parameters = CompletionParameters(
            model=test_model,
            instructions="",
            prompt="Write a two word poem.",
            max_tokens=100,
            temperature=0.8,
        )

        result = self.provider.provider_class_completion(completion_parameters)
        self.assertIsInstance(result, str)

    def test_provider_class_function_call(self):
        function_call_parameters = MagicMock()
        function_call_parameters.function = Function(
            name="test-function",
            description="test-description",
            arguments=[
                Argument(
                    name="arg1",
                    param_type=ParamType.STRING,
                    description="Return the letter a.",
                ),
                Argument(
                    name="arg2",
                    param_type=ParamType.INTEGER,
                    description="Return the number 0.",
                ),
            ],
        )

        mock_model = MagicMock()
        mock_model.generate_content.return_value = MockResponse(
            text='{"array": [{"arg1": "test-arg1", "arg2": 0}]}',
            candidates=[MockCandidate(MockFinishReason("STOP_SEQUENCE"))],
        )

        with patch(
            "llm_nexus.providers._googleai_provider.genai.GenerativeModel",
            return_value=mock_model,
        ):
            result = self.provider.provider_class_function_call(
                function_call_parameters
            )

        self.assertEqual(result, {"array": [{"arg1": "test-arg1", "arg2": 0}]})

    def test_real_function_call(self):
        function_call_parameters = FunctionCallParameters(
            user_prompt="Please return a random letter and a random integer.",
            function=Function(
                name="test-function",
                description="test-description",
                arguments=[
                    Argument(
                        name="Letter",
                        param_type=ParamType.STRING,
                        description="Return as random letter.",
                    ),
                    Argument(
                        name="Integer",
                        param_type=ParamType.INTEGER,
                        description="Return a random integer.",
                    ),
                ],
            ),
            model=test_model,
        )

        result = self.provider.provider_class_function_call(function_call_parameters)
        self.assertIsInstance(
            result, dict, f"Result is not a dictionary.\nResult: {result}"
        )
        log.debug(result)

    def test_provider_class_function_result_check(self):
        function = Function(
            name="test-function",
            description="test-description",
            arguments=[
                Argument(
                    name="arg1",
                    param_type=ParamType.STRING,
                    description="arg1-description",
                ),
                Argument(
                    name="arg2",
                    param_type=ParamType.INTEGER,
                    description="arg2-description",
                ),
            ],
        )
        actual_result = {"array": [{"arg1": "test-arg1", "arg2": 0}]}

        result = self.provider.provider_class_function_result_check(
            function, actual_result
        )
        self.assertTrue(result)

    def test_function_call_array(self):
        result = self.provider.provider_class_function_call(
            self.function_call_parameters_array,
            return_array=True,
        )
        log.debug(result)
        self.assertIsInstance(result, dict)
        # Assert the dict contains an "array" key
        self.assertIn("array", result)
        log.debug(result)

    def test_function_call_array_result_check(self):
        # Create a sample function
        function = Function(
            name="get_n_day_weather_forecast",
            description="Get an N-day weather forecast",
            arguments=[
                Argument(
                    name="location",
                    param_type="string",
                    description="The city and state, e.g. San Francisco, CA",
                ),
                Argument(
                    name="format",
                    param_type="string",
                    description="The temperature unit to use. Infer this from the users location.",
                ),
                Argument(
                    name="num_days",
                    param_type="integer",
                    description="The number of days to forecast",
                ),
            ],
        )

        # Create a sample result
        actual_result = {
            "array": [
                {
                    "location": "San Francisco, CA",
                    "format": "fahrenheit",
                    "num_days": 5,
                },
                {
                    "location": "New York, NY",
                    "format": "celsius",
                    "num_days": 7,
                },
                {
                    "location": "Los Angeles, CA",
                    "format": "fahrenheit",
                    "num_days": 3,
                },
            ]
        }

        # Call the method being tested
        result = self.provider.provider_class_function_result_check(
            function, actual_result
        )

        # Assert the result matches the expected result
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
