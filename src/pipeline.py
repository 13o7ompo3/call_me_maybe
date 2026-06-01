from .loader import load_functions, load_prompts, save_results
from .vocab import Vocabulary
from .cache import RouterCache
from .generator import RoutingGenerator
from .extractor import ExtractionGenerator
from .hints import ARGUMENT_HINTS


class FunctionCallingPipeline:
    """End-to-end pipeline for LLM-based function calling.

    Attributes:
        functions_definition_path: Path to the function definitions JSON.
        input_path: Path to the prompts JSON.
        output_path: Path for the output JSON.
    """

    def __init__(
        self,
        functions_definition_path: str,
        input_path: str,
        output_path: str,
    ) -> None:
        """Initialise the pipeline with file paths.

        Args:
            functions_definition_path: Path to functions_definition.json.
            input_path: Path to function_calling_tests.json.
            output_path: Destination for function_calls.json.
        """
        self.functions_definition_path = functions_definition_path
        self.input_path = input_path
        self.output_path = output_path

    def run(self) -> None:
        functions = load_functions(self.functions_definition_path)
        prompts = load_prompts(self.input_path)
        functions_map = {f.name: f for f in functions}

        vocab = Vocabulary()
        cache = RouterCache(vocab.id_to_token, list(functions_map.keys()))

        router = RoutingGenerator(vocab.llm, vocab)
        extractor = ExtractionGenerator(vocab.llm, vocab, ARGUMENT_HINTS)

        # 3. The Batch Pipeline
        results = []

        print(f"\n--- Processing {len(prompts)} Test Cases ---")
        for i, test_case in enumerate(prompts):
            print(f"\n[{i+1}/{len(prompts)}] Processing: {test_case.prompt}")

            # Phase 1: Function name extraction
            chosen_func_name = router.route(test_case.prompt, cache, functions)

            # Phase 2: Argument extraction
            extracted_args = {}
            if chosen_func_name != "fn_unknown":
                extracted_args = extractor.extract(test_case.prompt,
                                                   chosen_func_name,
                                                   functions_map[
                                                       chosen_func_name])

            final_json = {
                "prompt": test_case.prompt,
                "name": chosen_func_name,
                "parameters": extracted_args
            }
            results.append(final_json)

        # 4. Save to Disk
        save_results(self.output_path, results)
