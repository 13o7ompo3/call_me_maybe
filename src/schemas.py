from pydantic import BaseModel, Field, ValidationError
from typing import Dict, List, Literal, Any


# --- 1. Schemas for functions_definition.json ---
class ParameterDef(BaseModel):
    type: Literal["string", "number", "boolean", "integer"]


class ReturnDef(BaseModel):
    type: Literal["string", "number", "boolean", "integer"]


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, ParameterDef]
    returns: ReturnDef


# --- 2. Schema for function_calling_tests.json ---
class TestCase(BaseModel):
    prompt: str


# --- 3. Schema for the Final Output ---
class FunctionCallOutput(BaseModel):
    prompt: str
    name: str
    parameters: Dict[str, Any]
