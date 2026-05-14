import numpy as np
from typing import List
from llm_sdk import Small_LLM_Model
from .vocab import Vocabulary
from .fsm import JSONStateMachine, State

class ConstrainedGenerator:
    def __init__(self, llm: Small_LLM_Model, vocab: Vocabulary):
        self.llm = llm
        self.vocab = vocab

    def generate(self, prompt: str, fsm: JSONStateMachine, max_tokens: int = 100) -> str:
        print(f"\n[GENERATION START] Prompt: '{prompt}'")
        input_tensor = self.llm.encode(prompt)
        current_ids: List[int] = input_tensor.tolist()[0]

        generated_text = ""
        emitted_chunk = ""

        for step in range(max_tokens):
            logits = self.llm.get_logits_from_input_ids(current_ids)

            allowed_ids = fsm.get_valid_token_ids(emitted_chunk)

            if not allowed_ids:
                print("\n[GENERATION HALTED] FSM returned no valid tokens.")
                break

            mask = np.full(len(logits), -np.inf)

            for valid_id in allowed_ids:
                mask[valid_id] = logits[valid_id]

            next_token_id = int(np.argmax(mask))

            next_token_str = self.vocab.id_to_token[next_token_id]
            clean_str = next_token_str.replace("Ġ", " ").replace("\u0120", " ")

            current_ids.append(next_token_id)
            generated_text += clean_str
            emitted_chunk += clean_str

            print(clean_str, end="", flush=True)

            if self._advance_state(fsm, emitted_chunk):
                emitted_chunk = ""
            self._advance_state(fsm, emitted_chunk)

        print("\n[GENERATION COMPLETE]")
        return generated_text

    def _advance_state(self, fsm: JSONStateMachine, emitted_chunk: str) -> bool:
        clean_chunk = emitted_chunk.lstrip()

        if fsm.state == State.EXPECT_OPEN_BRACE and "{" in clean_chunk:
            fsm.state = State.EXPECT_NAME_KEY
            return True

        elif fsm.state == State.EXPECT_NAME_KEY and '"name":' in clean_chunk:
            fsm.state = State.EXPECT_FUNCTION_NAME
            return True

        elif fsm.state == State.EXPECT_FUNCTION_NAME:
            for name in fsm.allowed_functions:
                if clean_chunk == f'"{name}"':
                    fsm.active_function = name
                    fsm.state = State.EXPECT_PARAMS_KEY
                    return True

        elif fsm.state == State.EXPECT_PARAMS_KEY and ',"parameters":{' in clean_chunk:
            func_def = fsm.functions_map[fsm.active_function]
            fsm.remaining_params = list(func_def.parameters.keys())

            if fsm.remaining_params:
                fsm.state = State.EXPECT_PARAM_KEY
            else:
                fsm.state = State.EXPECT_PARAM_COMMA_OR_CLOSE
            return True

        elif fsm.state == State.EXPECT_PARAM_KEY:
            target = f'"{fsm.remaining_params[0]}"'
            if clean_chunk == target:
                fsm.state = State.EXPECT_PARAM_COLON
                return True

        elif fsm.state == State.EXPECT_PARAM_COLON and ":" in clean_chunk:
            current_param = fsm.remaining_params.pop(0)
            func_def = fsm.functions_map[fsm.active_function]
            fsm.current_param_type = func_def.parameters[current_param].type

            fsm.state = State.EXPECT_PARAM_VALUE
            return True

        elif fsm.state == State.EXPECT_PARAM_VALUE:
            if clean_chunk.endswith(",") or clean_chunk.endswith("}"):
                if len(fsm.remaining_params) > 0:
                    fsm.state = State.EXPECT_PARAM_KEY
                else:
                    fsm.state = State.EXPECT_PARAM_COMMA_OR_CLOSE
                return True

        return False
