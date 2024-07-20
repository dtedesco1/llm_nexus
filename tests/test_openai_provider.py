import os
import unittest

from src.llm_nexus._logging.log import log
from src.llm_nexus._utils.completions import CompletionParameters
from src.llm_nexus._utils.function_calls import (
    Argument,
    Function,
    FunctionCallParameters,
)
from src.llm_nexus.providers._openai_provider import OpenAIProvider


class TestOpenAIProvider(unittest.TestCase):
    def setUp(self):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.provider = OpenAIProvider(self.api_key)

        self.completion_parameters = CompletionParameters(
            instructions="Return a rhyme.",
            prompt="Would a rose by any other name smell as sweet?",
            model="gpt-3.5-turbo",
            temperature=0.8,
            max_tokens=100,
        )
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
            model="gpt-3.5-turbo",
            temperature=0.8,
        )

        # Write a call to return an array of historical events.
        self.function_call_parameters_array = FunctionCallParameters(
            user_prompt="What are three underrated historical events?",
            function=self.function,
            return_array=True,
            model="gpt-3.5-turbo",
            temperature=0.8,
        )

    def test_provider_class_completion(self):
        result = self.provider.provider_class_completion(self.completion_parameters)
        log.debug(result)

    def test_provider_class_function_call(self):
        result = self.provider.provider_class_function_call(
            self.function_call_parameters
        )
        log.debug(result)

    def test_format_function_call(self):
        # Create a sample list of arguments
        arguments = [
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
        ]
        # Create a sample function
        function = Function(
            name="get_n_day_weather_forecast",
            description="Get an N-day weather forecast",
            arguments=arguments,
        )

        # Call the method being tested
        result = self.provider._format_function_call(function, return_array=False)

        # Define the expected result
        expected_result = [
            {
                "type": "function",
                "function": {
                    "name": "get_n_day_weather_forecast",
                    "description": "Get an N-day weather forecast",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "format": {
                                "type": "string",
                                "description": "The temperature unit to use. Infer this from the users location.",
                            },
                            "num_days": {
                                "type": "integer",
                                "description": "The number of days to forecast",
                            },
                        },
                        "required": ["location", "format", "num_days"],
                    },
                },
            }
        ]

        # Assert the result matches the expected result
        self.assertEqual(result, expected_result)

    def test_provider_class_function_result_check(self):
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
            "location": "San Francisco, CA",
            "format": "fahrenheit",
            "num_days": 5,
        }

        # Call the method being tested
        result = self.provider.provider_class_function_result_check(
            function, actual_result
        )

        # Assert the result matches the expected result
        self.assertTrue(result)

    def test_function_call_array(self):
        result = self.provider.provider_class_function_call(
            self.function_call_parameters_array,
            return_array=True,
        )
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

    def test_real_function_call(self):
        # Create a sample list of arguments
        arguments = [
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
        ]
        # Create a sample function
        function = Function(
            name="get_n_day_weather_forecast",
            description="Get an N-day weather forecast",
            arguments=arguments,
        )

        # Create a sample function call
        function_call_parameters = FunctionCallParameters(
            user_prompt="What is the weather forecast for San Francisco, CA?",
            function=function,
            model="gpt-3.5-turbo",
            temperature=0.8,
            max_tokens=100,
        )

        # Call the method being tested
        result = self.provider.provider_class_function_call(function_call_parameters)

        # Assert the result is a dictionary
        self.assertIsInstance(result, dict)
        log.debug(result)

    # def test_invalid_api_key(self):
    #     with self.assertRaises(AuthenticationError):
    #         _OpenAIProvider("invalid-api-key")


if __name__ == "__main__":
    unittest.main()
