import numpy as np
from typing import List
from pydantic import BaseModel, ConfigDict
from llm_sdk import Small_LLM_Model
from src.vocab import Vocabulary
from src.cache import RouterCache
from src.schemas import FunctionDefinition
from src.prompts import build_routing_prompt


class RoutingGenerator(BaseModel):
    """Generate a function name from a user query using constrained decoding.

    Args:
        llm (Small_LLM_Model): Language model wrapper used for tokenization and
            logits.
        vocab (Vocabulary): Vocabulary helper that maps token IDs to strings.

    Returns:
        None.

    Raises:
        None.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    llm: Small_LLM_Model
    vocab: Vocabulary

    def route(self, user_query: str, cache: RouterCache,
              functions: List[FunctionDefinition]) -> str:
        """Route a natural-language query to the most likely function name.

        Args:
            user_query (str): Natural-language query from the user.
            cache (RouterCache): Prefix cache that constrains valid next
                tokens.
            functions (List[FunctionDefinition]): Available function
                definitions to choose from.

        Returns:
            str: Selected function name, or ``fn_unsupported_action``
            when no uniquematch can be determined.

        Raises:
            None.
        """
        injected_prompt = build_routing_prompt(user_query, functions)
        input_tensor = self.llm.encode(injected_prompt)
        current_ids: List[int] = input_tensor.tolist()[0]

        probable_functions = {fn.name: fn.name for fn in functions}
        generated_text = ""

        print(f"\n- Prompt: '{user_query}'")
        print("\nFunction: ", end="", flush=True)

        while True:
            raw_logits = self.llm.get_logits_from_input_ids(current_ids)
            logits = np.array(raw_logits)
            allowed_ids = cache.get_valid_token_ids(generated_text)

            if not allowed_ids:
                break

            allowed_logits = logits[allowed_ids]
            exp_logits = np.exp(allowed_logits - np.max(allowed_logits))
            probs = exp_logits / np.sum(exp_logits)
            next_token_idx = np.argmax(probs)
            next_token_id = allowed_ids[next_token_idx]
            next_token_prob = probs[next_token_idx]

            if next_token_prob < 0.6:
                break

            next_token_str = self.vocab.id_to_token[next_token_id]
            clean_str = next_token_str.replace(
                "Ġ", " ").replace("Ċ", "\n")

            current_ids.append(next_token_id)
            generated_text += clean_str
            print(clean_str, end="", flush=True)

            for fn_name, remain in probable_functions.copy().items():
                if remain.startswith(clean_str):
                    probable_functions[fn_name] = remain[len(clean_str):]
                else:
                    del probable_functions[fn_name]
            if len(probable_functions) == 1:
                return list(probable_functions.keys())[0]
            if len(probable_functions) == 0:
                break

        return "fn_unsupported_action"
