import unittest

from pydantic import ValidationError

from src.llm_nexus._utils.function_calls import (
    Argument,
    Function,
    FunctionCallParameters,
    ParamType,
)


class TestFunctionCallParameters(unittest.TestCase):
    def setUp(self):
        self.function = Function(
            name="Test function",
            description="Test description",
            arguments=[
                Argument(
                    name="Test argument",
                    description="Test description",
                    param_type=ParamType.STRING,
                )
            ],
        )

    def test_valid_input(self):
        # Test that FunctionCallParameters correctly validates its input
        params = FunctionCallParameters(
            user_prompt="Test user prompt",
            function=self.function,
            model="Test model",
        )
        self.assertEqual(params.user_prompt, "Test user prompt")
        self.assertEqual(params.function, self.function)
        self.assertEqual(params.model, "Test model")
        self.assertEqual(params.temperature, 0.8)  # default value

    def test_invalid_input(self):
        # Test that FunctionCallParameters raises an error for invalid input
        with self.assertRaises(ValidationError):
            FunctionCallParameters(
                user_prompt="Test user prompt",
                function="Invalid function",  # should be a Function instance
                model="Test model",
            )


class TestFunction(unittest.TestCase):
    def setUp(self):
        self.arguments = [
            Argument(
                name="Test argument",
                description="Test description",
                param_type=ParamType.STRING,
            )
        ]

    def test_valid_input(self):
        # Test that Function correctly validates its input
        function = Function(
            name="Test function",
            description="Test description",
            arguments=self.arguments,
        )
        self.assertEqual(function.name, "Test function")
        self.assertEqual(function.description, "Test description")
        self.assertEqual(function.arguments, self.arguments)

    def test_invalid_input(self):
        # Test that Function raises an error for invalid input
        with self.assertRaises(ValidationError):
            Function(
                name="Test function",
                description="Test description",
                arguments="Test arguments",  # should be a list of Argument instances
            )


class TestArgument(unittest.TestCase):
    def test_valid_input(self):
        # Test that Argument correctly validates its input
        argument = Argument(
            name="Test argument",
            description="Test description",
            param_type=ParamType.STRING,
        )
        self.assertEqual(argument.name, "Test argument")
        self.assertEqual(argument.description, "Test description")
        self.assertEqual(argument.param_type, ParamType.STRING)

    def test_invalid_input(self):
        # Test that Argument raises an error for invalid input
        with self.assertRaises(ValidationError):
            Argument(
                name="Test argument",
                description="Test description",
                param_type=1,  # should be a ParamType instance
            )


if __name__ == "__main__":
    unittest.main()
