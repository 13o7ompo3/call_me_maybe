import sys
from pathlib import Path
from src.loader import load_functions, load_prompts
from src.vocab import Vocabulary


def main() -> None:
    print("Call Me Maybe Engine: Initialized.\n")

    # 1. Load the Data Shield
    functions_path = Path("data/input/functions_definition.json")
    prompts_path = Path("data/input/function_calling_tests.json")

    functions = load_functions(functions_path)
    prompts = load_prompts(prompts_path)
    print(f"[SUCCESS] Loaded {len(functions)} functions and {len(prompts)} prompts.\n")

    # 2. Boot the Engine
    vocab = Vocabulary()
    print(f"[SUCCESS] Loaded {len(vocab.token_to_id)} tokens into memory.\n")

    # 3. Prove we have the lookup table working!
    crucial_chars = ["{", "}", '"name"', "Ġ{"] 
    print("--- Critical Token IDs ---")
    for char in crucial_chars:
        tok_id = vocab.get_id(char)
        print(f"Token '{char}' -> ID: {tok_id}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(1)
