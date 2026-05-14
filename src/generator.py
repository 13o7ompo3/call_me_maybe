import numpy as np
from typing import List
from llm_sdk import Small_LLM_Model
from .vocab import Vocabulary
from .fsm import JSONStateMachine, State
from .schemas import FunctionDefinition

def build_injected_prompt(user_prompt: str, functions: List[FunctionDefinition]) -> str:
    injected = 'You are a function calling system. Choose the correct function to answer the user prompt.\n\n'
    for func in functions:
        injected += f"Function: {func.name}\nDescription: {func.description}\n"
    injected += f"\nUser Prompt: {user_prompt}"
    return injected

class ConstrainedGenerator:
    def __init__(self, llm: Small_LLM_Model, vocab: Vocabulary):
        self.llm = llm
        self.vocab = vocab

    def generate(self, original_prompt: str, fsm: JSONStateMachine, functions: List[FunctionDefinition]) -> str:
        injected_prompt = build_injected_prompt(original_prompt, functions)
        input_tensor = self.llm.encode(injected_prompt)
        current_ids: List[int] = input_tensor.tolist()[0]

        generated_text = ""
        emitted_chunk = "" 

        for step in range(150):
            logits = self.llm.get_logits_from_input_ids(current_ids)
            allowed_ids = fsm.get_valid_token_ids(emitted_chunk)

            if not allowed_ids:
                print(f"\n[GENERATION HALTED] FSM Trap! State: {fsm.state}, Chunk: '{emitted_chunk}'")
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

            remainder = self._advance_state(fsm, emitted_chunk)
            if remainder is not None:
                emitted_chunk = remainder # Keep the leftovers!
                
            if generated_text.strip().endswith("}") and fsm.state == State.EXPECT_PARAM_VALUE:
                 break

        print("\n[GENERATION COMPLETE]")
        return generated_text

    def _advance_state(self, fsm: JSONStateMachine, emitted_chunk: str) -> str | None:
        """
        Instead of returning True/False, we return the leftover string 
        that needs to be passed to the next state. Return None if no state shift.
        """
        clean_chunk = emitted_chunk.lstrip()

        if fsm.state == State.EXPECT_OPEN_BRACE:
            idx = clean_chunk.find("{")
            if idx != -1:
                fsm.state = State.EXPECT_NAME_KEY
                return clean_chunk[idx+1:].lstrip()

        elif fsm.state == State.EXPECT_NAME_KEY:
            target = '"name":'
            if clean_chunk.startswith(target):
                fsm.state = State.EXPECT_FUNCTION_NAME
                return clean_chunk[len(target):].lstrip()

        elif fsm.state == State.EXPECT_FUNCTION_NAME:
            for name in fsm.allowed_functions:
                target = f'"{name}"'
                if clean_chunk.startswith(target):
                    fsm.active_function = name
                    fsm.state = State.EXPECT_PARAMS_KEY
                    return clean_chunk[len(target):].lstrip()

        elif fsm.state == State.EXPECT_PARAMS_KEY:
            target = ',"parameters":{'
            if clean_chunk.startswith(target):
                func_def = fsm.functions_map[fsm.active_function]
                fsm.remaining_params = list(func_def.parameters.keys())

                if fsm.remaining_params:
                    fsm.state = State.EXPECT_PARAM_KEY
                else:
                    fsm.state = State.EXPECT_PARAM_VALUE 
                return clean_chunk[len(target):].lstrip()

        elif fsm.state == State.EXPECT_PARAM_KEY:
            target = f'"{fsm.remaining_params[0]}"'
            if clean_chunk.startswith(target):
                fsm.state = State.EXPECT_PARAM_COLON
                return clean_chunk[len(target):].lstrip()

        elif fsm.state == State.EXPECT_PARAM_COLON:
            idx = clean_chunk.find(":")
            if idx != -1:
                current_param = fsm.remaining_params.pop(0)
                func_def = fsm.functions_map[fsm.active_function]
                fsm.current_param_type = func_def.parameters[current_param].type
                fsm.state = State.EXPECT_PARAM_VALUE
                return clean_chunk[idx+1:].lstrip()

        elif fsm.state == State.EXPECT_PARAM_VALUE:
            if clean_chunk.endswith(","):
                fsm.state = State.EXPECT_PARAM_KEY
                return ""
            elif clean_chunk.endswith("}"):
                return "" 

        return None