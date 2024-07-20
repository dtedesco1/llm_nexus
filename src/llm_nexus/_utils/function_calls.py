from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class FunctionCallParameters(BaseModel):
    """
    Represents the parameters for a function call.

    Attributes:
        user_prompt (str): The prompt for the user to follow.
        function (Function): The function to call.
        model (str): The model to use.
        max_tokens (int): The maximum number of tokens to generate.
        temperature (float): The temperature to use.
    """

    user_prompt: str
    function: Function
    model: str
    max_tokens: int = 100
    temperature: float = 0.8


class Function(BaseModel):
    """
    Represents a function to be passed to an LLM for function calling.

    Attributes:
        name (str): The name of the function.
        description (str): The description of the function.
        arguments (List[Argument]): The list of arguments for the function.
    """

    name: str
    description: str
    arguments: List[Argument]

    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "arguments": [arg.to_dict() for arg in self.arguments],
        }

    def expected_arguments(self):
        return [arg.name for arg in self.arguments]

    def arg_dict(self):
        return {arg.name: arg.description for arg in self.arguments}


class ParamType(Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"


class Argument(BaseModel):
    """
    Represents a parameter to be passed to an LLM for function calling.

    Attributes:
        name (str): The name of the parameter.
        param_type (ParamType): The type of the parameter.
        description (str): The description of the parameter.
    """

    name: str
    param_type: ParamType
    description: str

    def to_dict(self):
        return {
            "name": self.name,
            "type": self.param_type.value,
            "description": self.description,
        }

    def arg_format(self):
        return f"{self.name}: {self.description}"
