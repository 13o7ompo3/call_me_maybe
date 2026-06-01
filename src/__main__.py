import argparse
import sys
from .pipeline import FunctionCallingPipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(prog="python -m src")
    parser.add_argument(
        "--functions_definition",
        metavar="FUNCTIONS_DEFINITION_FILE_PATH",
        default="data/input/functions_definition.json",
        help="Path to the functions definition JSON file.",
    )
    parser.add_argument(
        "--input",
        metavar="INPUT_FILE_PATH",
        default="data/input/function_calling_tests.json",
        help="Path to the input prompts JSON file.",
    )
    parser.add_argument(
        "--output",
        metavar="OUTPUT_FILE_PATH",
        default="data/output/function_calls.json",
        help="Path to the output JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        pipeline = FunctionCallingPipeline(
            functions_definition_path=args.functions_definition,
            input_path=args.input,
            output_path=args.output,
        )
        pipeline.run()
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
