from .loader import (load_functions, load_prompts,
                     save_results, porobable_functions)
from .vocab import Vocabulary
from .cache import RouterCache
from .generator import RoutingGenerator
from .extractor import ExtractionGenerator
from .hints import ARGUMENT_HINTS
from pydantic import BaseModel


class FunctionCallingPipeline(BaseModel):
    """Run routing and argument extraction over a batch of test prompts.

    Args:
        functions_definition_path (str): Path to the function definitions JSON.
        input_path (str): Path to the prompts JSON file.
        output_path (str): Path where the results JSON will be written.

    Returns:
        None.

    Raises:
        None.
    """

    functions_definition_path: str
    input_path: str
    output_path: str

    def run(self) -> None:
        """Execute the full function-calling pipeline and persist results.

        Args:
            None.

        Returns:
            None.

        Raises:
            SystemExit: Propagated from the loading or saving helpers when they
            fail.
        """
        functions = load_functions(self.functions_definition_path)
        prompts = load_prompts(self.input_path)
        functions_map = porobable_functions(functions)

        vocab = Vocabulary()
        cache = RouterCache(vocab=vocab.id_to_token,
                            allowed_functions=list(functions_map.keys()))

        router = RoutingGenerator(llm=vocab.llm, vocab=vocab)
        extractor = ExtractionGenerator(llm=vocab.llm, vocab=vocab,
                                        hints=ARGUMENT_HINTS)

        # 3. The Batch Pipeline
        results = []

        print(f"\n--- Processing {len(prompts)} Test Cases ---")
        for i, test_case in enumerate(prompts):
            print(f"[{i+1}/{len(prompts)}] Processing: {test_case.prompt}")

            # Phase 1: Function name extraction
            chosen_func_name = router.route(test_case.prompt, cache, functions)
            print(f"\n\n  → function: {chosen_func_name}\n")

            # Phase 2: Argument extraction
            extracted_args = {}
            if chosen_func_name != "fn_unsupported_action":
                extracted_args = extractor.extract(test_case.prompt,
                                                   chosen_func_name,
                                                   functions_map[
                                                       chosen_func_name])
            print(f"\n  → arguments: {extracted_args}\n")

            final_json = {
                "prompt": test_case.prompt,
                "name": chosen_func_name,
                "parameters": extracted_args
            }
            results.append(final_json)

        # 4. Save to Disk
        save_results(self.output_path, results)
