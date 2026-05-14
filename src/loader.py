import json
import sys
from pathlib import Path
from pydantic import ValidationError
from src.schemas import FunctionDefinition, TestCase
from typing import Callable, Any


def error_and_exit(function: Callable[[Path], Any]) -> Callable[[Path], Any]:
    def wrapper(*args, **kwargs):
        try:
            filepath = args[0] if args else kwargs.get('filepath')
            return function(*args, **kwargs)
        except FileNotFoundError:
            print(f"Error: Required file not found at '{filepath}'.")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"Error: The file '{filepath}' "
                  "contains invalid JSON formatting.")
            sys.exit(1)
        except ValidationError as e:
            print(f"Error: The data in '{filepath}' "
                  "violates the required schema constraints.")
            print(e)
            sys.exit(1)
    return wrapper


@error_and_exit
def load_functions(filepath: Path) -> list[FunctionDefinition]:
    with open(filepath, 'r') as f:
        raw_data = json.load(f)
    return [FunctionDefinition(**item) for item in raw_data]


@error_and_exit
def load_prompts(filepath: Path) -> list[TestCase]:
    with open(filepath, 'r') as f:
        raw_data = json.load(f)
    return [TestCase(**item) for item in raw_data]
