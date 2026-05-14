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

    test_prompt = 'You are a function calling system. Choose the correct function to answer the user prompt.\
Available functions:\
    "name": "fn_add_numbers",\
    "description": "Add two numbers together and return their sum.",\
\
    "name": "fn_greet",\
    "description": "Generate a greeting message for a person by name.",\
\
    "name": "fn_reverse_string",\
    "description": "Reverse a string and return the reversed result.",\
\
    "name": "fn_get_square_root",\
    "description": "Calculate the square root of a number.",\
\
    "name": "fn_substitute_string_with_regex",\
    "description": "Replace all occurrences matching a regex pattern in a string.",\
User Prompt: What is the sum of 2 and 3?\
JSON Output:'

    generator.generate(test_prompt, fsm)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
