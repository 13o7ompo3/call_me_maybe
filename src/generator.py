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

        generated_text = "fn_"

        print(f"\n[ROUTING] Query: '{user_query}'")
        print("fn_", end="", flush=True)

        for step in range(15):
            logits = self.llm.get_logits_from_input_ids(current_ids)
            allowed_ids = cache.get_valid_token_ids(generated_text)

            if not allowed_ids:
                print("\n[ROUTING ERROR] Model hit a dead end at: "
                      f"'{generated_text}'")
                break

            mask = np.full(len(logits), -np.inf)
            for valid_id in allowed_ids:
                mask[valid_id] = logits[valid_id]

            next_token_id = int(np.argmax(mask))
            next_token_str = self.vocab.id_to_token[next_token_id]
            clean_str = next_token_str.replace("Ġ", " ").replace("\u0120", " ")

            current_ids.append(next_token_id)
            generated_text += clean_str
            print(clean_str, end="", flush=True)

            if generated_text in cache.allowed_functions:
                print("\n[ROUTING COMPLETE] Target Acquired.")
                return generated_text

        return generated_text
