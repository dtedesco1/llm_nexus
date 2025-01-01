import os
import unittest

from src.llm_nexus._logging.log import log
from src.llm_nexus._utils.function_calls import (
    Argument,
    Function,
    FunctionCallParameters,
    ParamType,
)
from src.llm_nexus.model_provider import ModelProvider

test_providers = ["openai", "anthropic", "googleai"]
test_models = ["gpt-3.5-turbo", "claude-3-5-haiku-20241022", "gemini-1.5-pro-latest"]

# # Test for OpenAI only
# test_providers = ["openai"]
# test_models = ["gpt-3.5-turbo"]

# # Test for Anthropic only
# test_providers = ["anthropic"]
# test_models = ["claude-3-5-haiku-20241022"]

# # For testing Google AI only
# test_providers = ["googleai"]
# test_models = ["gemini-1.5-pro-latest"]


class TestModelProvider(unittest.TestCase):
    def setUp(self):
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.googleai_api_key = os.environ.get("GOOGLEAI_API_KEY")
        self.provider = ModelProvider(
            self.openai_api_key, self.anthropic_api_key, self.googleai_api_key
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
        self.function: Function = Function(
            name="UnderratedHistoricalEvent",
            description="Returns an underrated historical event.",
            arguments=[arg1, arg2],
        )

    def test_init(self):
        self.assertIn("openai", self.provider.providers)
        self.assertIn("anthropic", self.provider.providers)
        self.assertIn("googleai", self.provider.providers)

    def test_check_provider_setup(self):
        self.assertTrue(self.provider._check_provider_setup("openai"))
        self.assertTrue(self.provider._check_provider_setup("anthropic"))
        self.assertTrue(self.provider._check_provider_setup("googleai"))
        self.assertFalse(self.provider._check_provider_setup("invalid_provider"))

    def test_completion(self):
        for provider, model in zip(test_providers, test_models):
            result = self.provider.completion(
                instructions="Return a rhyme.",
                user_prompt="Would a rose by any other name smell as sweet?",
                provider_name=provider,
                model_name=model,
                temperature=0.8,
                max_tokens=50,
            )
            log.debug(f"{model} Completion Result:\n{result}\n")
            self.assertIsInstance(result, str)

    def test_function_call(self):
        for provider, model in zip(test_providers, test_models):
            result = self.provider.function_call(
                user_prompt="What is an underrated historical event?",
                function=self.function,
                return_array=False,
                provider_name=provider,
                model_name=model,
                temperature=0.8,
                max_tokens=500,
            )
            self.assertIsInstance(
                result,
                dict,
                f"`test_function_call` result for {provider} is not a list: {result}",
            )
            # Assert `result` includes an 'array' key
            self.assertIn("array", result)
            # Assert `result['array']` is a list
            self.assertIsInstance(result["array"], list)


if __name__ == "__main__":
    unittest.main()
