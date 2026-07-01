import json
import sys
from pathlib import Path
from pydantic import ValidationError
from src.schemas import FunctionDefinition, TestCase, ReturnDef
from typing import Callable, Any, Dict


def error_and_exit(function: Callable[[str], list[Any]]
                   ) -> Callable[[str], list[Any]]:
    """Wrap a loader function so file and schema errors exit cleanly.

    Args:
        function (Callable[[str], Any]): Loader function that accepts a file
            path and returns parsed data.

    Returns:
        Callable[[str], Any]: Wrapper that prints a user-facing error and
        exits with status 1 on failure.

    Raises:
        None.
    """
    def wrapper(*args: Any, **kwargs: Dict[str, Any]) -> Any:
        """Execute the wrapped loader and normalize common failures.

        Args:
            *args (Any): Positional arguments forwarded to the wrapped loader.
            **kwargs (Dict[str, Any]): Keyword arguments forwarded to the
                wrapped loader.

        Returns:
            Any: The wrapped function result.

        Raises:
            SystemExit: When the file is missing, JSON is invalid, or schema
            validation fails.
        """
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
def load_functions(filepath: str) -> list[FunctionDefinition]:
    """Load function definitions from a JSON file.

    Args:
        filepath (str): Path to the JSON file containing function definitions.

    Returns:
        list[FunctionDefinition]: Parsed function definition objects.

    Raises:
        SystemExit: If the file is missing, invalid JSON, or fails validation.
    """
    with open(filepath, 'r') as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list):
        print("Error: Expected a list of function definitions in "
              f"'{filepath}'.")
        sys.exit(1)
    if not raw_data:
        print(f"Error: No function definitions found in '{filepath}'.")
        sys.exit(1)
    return [FunctionDefinition(**item) for item in raw_data]


@error_and_exit
def load_prompts(filepath: str) -> list[TestCase]:
    """Load function-calling prompts from a JSON file.

    Args:
        filepath (str): Path to the JSON file containing test prompts.

    Returns:
        list[TestCase]: Parsed prompt objects.

    Raises:
        SystemExit: If the file is missing, invalid JSON, or fails validation.
    """
    with open(filepath, 'r') as f:
        raw_data = json.load(f)
    if not isinstance(raw_data, list):
        print(f"Error: Expected a list of test cases in '{filepath}'.")
        sys.exit(1)
    if not raw_data:
        print(f"Error: No test cases found in '{filepath}'.")
        sys.exit(1)
    return [TestCase(**item) for item in raw_data]


def save_results(filepath: str, data: list[Dict[str, Any]]) -> None:
    """Write pipeline results to a JSON file.

    Args:
        filepath (str): Destination file path for the JSON output.
        data (list[Dict[str, Any]]): Result rows to serialize.

    Returns:
        None.

    Raises:
        SystemExit: If the output file cannot be written.
    """
    path = Path(filepath).parent
    path.mkdir(parents=True, exist_ok=True)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error: Failed to save results to '{filepath}'.")
        print(e)
        sys.exit(1)


def porobable_functions(functions: list[FunctionDefinition]
                        ) -> Dict[str, FunctionDefinition]:
    """Create a mapping of function names to themselves for tracking.

    Args:
        functions (list[FunctionDefinition]): List of function definitions.
    Returns:
        Dict[str, str]: Mapping of function names to themselves.
                and additionally add a reserved entry for unsupported actions.
    """
    unsuported_fn = FunctionDefinition(
            name="fn_unsupported_action",
            description="Triggers ONLY when the user query "
            "does not fit any other function. Use this for general questions, "
            "conversational chitchat, weather, philosophy, math calculations.",
            parameters={},
            returns=ReturnDef(type="string"))
    functions_map = {}
    for func in functions:
        if func.name in functions_map:
            print(f"Error: Duplicate function name '{func.name}' found.")
            sys.exit(1)
        functions_map[func.name] = func
    if "fn_unsupported_action" in functions_map:
        print("Warning: 'fn_unsupported_action' is reserved for "
              "unsupported queries. It will be overridden.")
    functions_map["fn_unsupported_action"] = unsuported_fn
    functions.append(unsuported_fn)
    return functions_map
