from enum import Enum, auto
from typing import Dict, List


class State(Enum):
    EXPECT_OPEN_BRACE = auto()
    EXPECT_NAME_KEY = auto()
    EXPECT_FUNCTION_NAME = auto()
    EXPECT_PARAMS_KEY = auto()
    EXPECT_ARGUMENTS = auto()


class JSONStateMachine:
    def __init__(self, vocab_map: Dict[int, str], allowed_functions: List[str]):
        print("Pre-computing AOT vocabulary masks...")
        self.vocab = vocab_map
        self.allowed_functions = allowed_functions
        self.state = State.EXPECT_OPEN_BRACE

        self.cache_open_brace = self._find_tokens_starting_with("{")
        self.cache_colon = self._find_tokens_starting_with(":")

        self.cache_name_key = self._build_prefix_cache('"name":')
        self.cache_params_key = self._build_prefix_cache(',"parameters":{')
        self.cache_function_names = self._build_functions_cache(allowed_functions)

        print("[SUCCESS] FSM Caches built and ready.")

    def _find_tokens_starting_with(self, target: str) -> List[int]:
        valid_ids = []
        for tid, tok_str in self.vocab.items():
            clean = tok_str.replace("Ġ", " ").replace("\u0120", " ").lstrip()
            if clean.startswith(target) and clean:
                valid_ids.append(tid)
        return valid_ids

    def _build_prefix_cache(self, target: str) -> Dict[str, List[int]]:
        """
        Calculates allowed next tokens for EVERY possible partial state of a target string.
        """
        cache: Dict[str, List[int]] = {}
        for i in range(len(target)):
            already_emitted = target[:i]
            remaining = target[i:]
            valid_ids = []

            for tid, tok_str in self.vocab.items():
                clean = tok_str.replace("Ġ", " ").replace("\u0120", " ").lstrip()
                if not clean:
                    continue

                if remaining.startswith(clean) or clean.startswith(remaining):
                    valid_ids.append(tid)

            cache[already_emitted] = valid_ids
        return cache

    def _build_functions_cache(self, allowed_functions: List[str]) -> Dict[str, List[int]]:
        """Pre-computes masks for every possible valid function name, wrapped in quotes."""
        master_cache: Dict[str, List[int]] = {}
        for func_name in allowed_functions:
            target = f' "{func_name}"' 
            func_cache = self._build_prefix_cache(target)
            
            for prefix, ids in func_cache.items():
                if prefix not in master_cache:
                    master_cache[prefix] = []
                master_cache[prefix].extend(ids)
                master_cache[prefix] = list(set(master_cache[prefix]))
                
        return master_cache

    def _get_already_emitted(self, target: str, emitted_text: str) -> str:
        """Finds how much of the target string we have successfully output so far."""
        for i in range(min(len(target), len(emitted_text)), 0, -1):
            if emitted_text.endswith(target[:i]):
                return target[:i]
        return ""

    def get_valid_token_ids(self, emitted_chunk: str) -> List[int]:
        if self.state == State.EXPECT_OPEN_BRACE:
            return self.cache_open_brace
            
        elif self.state == State.EXPECT_NAME_KEY:
            return self.cache_name_key.get(emitted_chunk, [])
            
        elif self.state == State.EXPECT_FUNCTION_NAME:
            return self.cache_function_names.get(emitted_chunk, [])
            
        elif self.state == State.EXPECT_PARAMS_KEY:
            return self.cache_params_key.get(emitted_chunk, [])
        return []
