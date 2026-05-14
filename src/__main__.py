import sys
from pathlib import Path
from .loader import load_functions, load_prompts
from .vocab import Vocabulary
from .fsm import JSONStateMachine
from .generator import ConstrainedGenerator


def main() -> None:
    print("Call Me Maybe Engine: Initialized.\n")

    functions = load_functions(Path("data/input/functions_definition.json"))
    vocab = Vocabulary()

    fsm = JSONStateMachine(vocab.id_to_token, functions)

    generator = ConstrainedGenerator(vocab.llm, vocab)

    test_prompt = "What is the sum of 2 and 3?"

    generator.generate(test_prompt, fsm, functions)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
