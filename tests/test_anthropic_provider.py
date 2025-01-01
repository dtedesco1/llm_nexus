import os
import unittest

from src.llm_nexus._logging.log import log
from src.llm_nexus._utils.completions import CompletionParameters
from src.llm_nexus._utils.function_calls import (
    Argument,
    Function,
    FunctionCallParameters,
    ParamType,
)
from src.llm_nexus.providers._anthropic_provider import AnthropicProvider

test_model = "claude-3-5-haiku-20241022"


class TestAnthropicProvider(unittest.TestCase):
    def setUp(self):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.provider = AnthropicProvider(api_key=api_key)

        self.completion_parameters = CompletionParameters(
            instructions="Return a rhyme.",
            prompt="Would a rose by any other name smell as sweet?",
            model=test_model,
            temperature=0.8,
            max_tokens=100,
        )
        # Create Argument objects
        arg1 = Argument(
            name="Year",
            param_type=ParamType.INTEGER,
            description="The year of the event.",
        )
        arg2 = Argument(
            name="Event",
            param_type=ParamType.STRING,
            description="The name of the event.",
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
        )

    def test_provider_class_function_call_with_array(self):
        function = Function(
            name="underrated_historical_events",
            description="Name 3 underrated historical AD events based on the user's request",
            arguments=[
                Argument(
                    name="event",
                    param_type=ParamType.STRING,
                    description="The name of the event.",
                ),
                Argument(
                    name="year",
                    param_type=ParamType.INTEGER,
                    description="The starting year of the event.",
                ),
                Argument(
                    name="reasoning",
                    param_type=ParamType.STRING,
                    description="A brief description of why you feel this event is underrated.",
                ),
            ],
        )

        function_call_parameters = FunctionCallParameters(
            user_prompt="What are some underrated AD events in African history?",
            function=function,
            model=test_model,
            temperature=0.8,
        )

        result = self.provider.provider_class_function_call(
            function_call_parameters=function_call_parameters,
            return_array=True,
        )

        # Assert the result is not empty
        self.assertTrue(result)

    def test_provider_class_completion(self):
        result = self.provider.provider_class_completion(self.completion_parameters)
        # Assert the result is not empty
        self.assertTrue(result)
        log.debug(result)

    def test_provider_class_function_call(self):
        result = self.provider.provider_class_function_call(
            self.function_call_parameters,
        )
        # Assert the result is not empty
        self.assertTrue(result)
        print(result)
        log.debug(result)

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
            "array": [
                {
                    "location": "San Francisco, CA",
                    "format": "fahrenheit",
                    "num_days": 5,
                }
            ]
        }

        # Call the method being tested
        result = self.provider.provider_class_function_result_check(
            function, actual_result
        )

        # Assert the result matches the expected result
        self.assertTrue(result)

    # TODO Unable to catch this for now.
    # def test_invalid_api_key(self):
    #     with self.assertRaises(AuthenticationError):
    #         _AnthropicProvider("invalid-api-key")


if __name__ == "__main__":
    unittest.main()
