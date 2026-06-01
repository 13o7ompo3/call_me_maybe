import numpy as np
from typing import List
from llm_sdk import Small_LLM_Model
from src.vocab import Vocabulary
from src.cache import RouterCache
from src.schemas import FunctionDefinition
from src.prompts import build_routing_prompt


class RoutingGenerator:
    def __init__(self, llm: Small_LLM_Model, vocab: Vocabulary):
        self.llm = llm
        self.vocab = vocab

    def route(self, user_query: str, cache: RouterCache,
              functions: List[FunctionDefinition]) -> str:
        injected_prompt = build_routing_prompt(user_query, functions)
        input_tensor = self.llm.encode(injected_prompt)
        current_ids: List[int] = input_tensor.tolist()[0]

        probable_functions = {fn.name: fn.name for fn in functions}

        generated_text = ""

        print(f"\n- Prompt: '{user_query}'")
        print("ANALYSIS:", end="", flush=True)

        for step in range(40):
            logits = self.llm.get_logits_from_input_ids(current_ids)

            if "|" not in generated_text:
                next_token_id = int(np.argmax(logits))
            else:
                after_pipe = generated_text.split("|")[1].lstrip()
                allowed_ids = cache.get_valid_token_ids(after_pipe)

                if not allowed_ids:
                    break

                mask = np.full(len(logits), -np.inf)
                for valid_id in allowed_ids:
                    mask[valid_id] = logits[valid_id]

                next_token_id = int(np.argmax(mask))

            next_token_str = self.vocab.id_to_token[next_token_id]
            clean_str = next_token_str.replace(
                "Ġ", " ").replace("\u0120", " ").replace("Ċ", "\n")

            current_ids.append(next_token_id)
            generated_text += clean_str
            print(clean_str, end="", flush=True)

            if "|" in generated_text and not clean_str.strip() in ["", "|"]:
                for fn_name, remain in probable_functions.copy().items():
                    if remain.startswith(clean_str):
                        probable_functions[fn_name] = remain[len(clean_str):]
                    else:
                        del probable_functions[fn_name]
                if len(probable_functions) == 1:
                    return list(probable_functions.keys())[0]
                if len(probable_functions) == 0:
                    break
        return "fn_unknown"
