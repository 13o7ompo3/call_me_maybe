import sys
from pathlib import Path
from src.loader import load_functions, load_prompts
from src.vocab import Vocabulary
from src.generator import RoutingGenerator
from src.cache import RouterCache


def main() -> None:
    print("Call Me Maybe Engine: Initialized.\n")

    functions = load_functions(Path("data/input/functions_definition.json"))
    prompts = load_prompts(Path("data/input/function_calling_tests.json"))
    vocab = Vocabulary()

    for test in prompts:
        cache = RouterCache(vocab.id_to_token, [f.name for f in functions])
        generator = RoutingGenerator(vocab.llm, vocab)
        chosen_function = generator.route(test.prompt, cache, functions)
        print(f"Final Decision: '{chosen_function}'\n{'-'*50}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
