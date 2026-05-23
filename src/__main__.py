import sys
import json
import time
from pathlib import Path

from .loader import load_functions, load_prompts
from .vocab import Vocabulary
from .cache import RouterCache
from .generator import RoutingGenerator
from .extractor import ExtractionGenerator


ARGUMENT_HINTS = {
    "fn_square_root": {
        "number": "The value given from the user query that the function will calculate the square root of."
    },
    "fn_substitute_string_with_regex": {
        "source_string": "The exact target text to be modified, STRICTLY EXCLUDING instruction words like 'Replace' or 'Substitute'. It is usually the text inside the quotes.",
        "regex": "The mathematical search pattern. '[a-zA-Z]' for set of letters, '\\d+' for numbers, the exact character or word to be replaced.",
    },
    "fn_read_file": {
        "path": "The absolute, complete file path, including all directories and slashes /home/user/file.txt."
    },
    "fn_format_template": {
        "template": "The template provided by the user, strictly including the curly braces {}. example: 'Say Hello to {name}'"
    }
}


def main() -> None:
    print("Call Me Maybe Engine: Initialized.\n")

    # 1. Paths & Folders
    input_dir = Path("data/input")
    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    functions = load_functions(input_dir / "functions_definition.json")
    functions_map = {f.name: f for f in functions}
    prompts = load_prompts(input_dir / "function_calling_tests.json")

    # 2. Boot the Engines
    vocab = Vocabulary()
    cache = RouterCache(vocab.id_to_token, list(functions_map.keys()))

    router = RoutingGenerator(vocab.llm, vocab)
    extractor = ExtractionGenerator(vocab.llm, vocab, ARGUMENT_HINTS)

    # 3. The Batch Pipeline
    results = []
    start_time = time.time()

    print(f"\n--- Processing {len(prompts)} Test Cases ---")
    for i, test_case in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] Query: {test_case.prompt}")

        # Phase 1: High-Speed Mathematical Routing
        chosen_func_name = router.route(test_case.prompt, cache, functions)

        # Phase 2: XML Data Extraction
        extracted_args = {}
        if chosen_func_name != "fn_unknown":
            extracted_args = extractor.extract(test_case.prompt,
                                               chosen_func_name,
                                               functions_map[chosen_func_name])

        # Phase 3: Assembly
        final_json = {
            "prompt": test_case.prompt,
            "name": chosen_func_name,
            "parameters": extracted_args
        }
        results.append(final_json)

    # 4. Save to Disk
    output_path = output_dir / "function_calling_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\n=========================================")
    print(f"[SUCCESS] Engine Shutdown.")
    print(f"Processed {len(prompts)} prompts in {elapsed:.2f} seconds.")
    print(f"File saved to: {output_path}")
    print(f"=========================================")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[ABORT] Process killed by user.")
        sys.exit(1)
