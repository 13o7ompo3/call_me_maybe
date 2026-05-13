import json
import sys
from pathlib import Path
from typing import List
from pydantic import ValidationError
from src.schemas import FunctionDefinition, TestCase


def load_function_definitions(filepath: Path) -> List[FunctionDefinition]:
    try:
        with open(filepath, 'r') as f:
            raw_data = json.load(f)

        # Validate the entire list of dictionaries against our strict schema
        return [FunctionDefinition(**item) for item in raw_data]

    except FileNotFoundError:
        print(f"Error: Could not find definitions file at {filepath}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: The file {filepath} contains invalid JSON formatting.")
        sys.exit(1)
    except ValidationError as e:
        print(f"Error: Data in {filepath} violates the required schema!")
        print(e)
        sys.exit(1)


def load_test_cases(filepath: Path) -> List[TestCase]:
    # Similar logic for loading the prompts
    try:
        with open(filepath, 'r') as f:
            raw_data = json.load(f)
        return [TestCase(**item) for item in raw_data]
    except (FileNotFoundError, json.JSONDecodeError, ValidationError) as e:
        print(f"Failed to load test cases: {e}")
        sys.exit(1)


def main():
    definitions = load_function_definitions(
        Path("./src/data/input/functions_definition.json"))
    allowed_names = [func.name for func in definitions]
    print(allowed_names)


if __name__ == "__main__":
    main()
