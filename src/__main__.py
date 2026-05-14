import sys
from pathlib import Path
from .loader import load_functions, load_prompts
from .vocab import Vocabulary
from .fsm import JSONStateMachine


def main() -> None:
    print("Call Me Maybe Engine: Initialized.\n")

    functions = load_functions(Path("data/input/functions_definition.json"))
    vocab = Vocabulary()

    allowed_names = [f.name for f in functions]

    fsm = JSONStateMachine(vocab.id_to_token, allowed_names)

    print("\n--- Cache Verification ---")

    partial_string = '"na'
    allowed_next_ids = fsm.cache_name_key.get(partial_string, [])

    print(f"If generated text is '{partial_string}', allowed next tokens are:")
    for tid in allowed_next_ids[:5]: # Just print the first 5 to avoid spam
        print(f" -> ID: {tid} ('{vocab.id_to_token[tid]}')")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
