from pydantic import BaseModel


class CompletionParameters(BaseModel):
    """
    Represents the parameters for a generic text completion.

    Attributes:
        instructions (str): The instructions for the model to follow.
        prompt (str): The prompt for the model to follow.
        model (str): The model to use.
        temperature (float): The temperature to use.
        max_tokens (int): The maximum number of tokens to generate.
    """

    instructions: str
    prompt: str
    model: str
    temperature: float = 0.8
    max_tokens: int = 100
