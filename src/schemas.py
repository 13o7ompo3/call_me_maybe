from pydantic import BaseModel
from typing import Dict, Literal, Any


# --- 1. Schemas for functions_definition.json ---
class ParameterDef(BaseModel):
    """Describe a single function parameter in the exported schema.

    Args:
        type (Literal["string", "number", "boolean", "integer"]): Type label
            for the parameter.

    Returns:
        None.

    Raises:
        None.
    """

    type: Literal["string", "number", "boolean", "integer"]


class ReturnDef(BaseModel):
    """Describe a function return value in the exported schema.

    Args:
        type (Literal["string", "number", "boolean", "integer"]): Type label
            for the return value.

    Returns:
        None.

    Raises:
        None.
    """

    type: Literal["string", "number", "boolean", "integer"]


class FunctionDefinition(BaseModel):
    """Describe a function available to the routing pipeline.

    Args:
        name (str): Function name used by the router.
        description (str): Natural-language summary of the function.
        parameters (Dict[str, ParameterDef]): Parameter schema mapping.
        returns (ReturnDef): Return-value schema.

    Returns:
        None.

    Raises:
        None.
    """

    name: str
    description: str
    parameters: Dict[str, ParameterDef]
    returns: ReturnDef


# --- 2. Schema for function_calling_tests.json ---
class TestCase(BaseModel):
    """Represent a single prompt used for function-calling evaluation.

    Args:
        prompt (str): Natural-language prompt to evaluate.

    Returns:
        None.

    Raises:
        None.
    """

    prompt: str


# --- 3. Schema for the Final Output ---
class FunctionCallOutput(BaseModel):
    """Represent one predicted function call and its arguments.

    Args:
        prompt (str): Original user prompt.
        name (str): Predicted function name.
        parameters (Dict[str, Any]): Parsed argument values.

    Returns:
        None.

    Raises:
        None.
    """

    prompt: str
    name: str
    parameters: Dict[str, Any]
