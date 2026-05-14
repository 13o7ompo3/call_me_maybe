from enum import Enum, auto
from typing import Dict, List


class State(Enum):
    EXPECT_OPEN_BRACE = auto()
    EXPECT_NAME_KEY = auto()
    EXPECT_FUNCTION_NAME = auto()
    EXPECT_PARAMS_KEY = auto()


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

        print("[SUCCESS] FSM Caches built and ready.")

    def _find_tokens_starting_with(self, target: str) -> List[int]:
        valid_ids = []
        for tid, tok_str in self.vocab.items():
            # Clean the special Qwen space characters
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

    def _get_already_emitted(self, target: str, emitted_text: str) -> str:
        """Finds how much of the target string we have successfully output so far."""
        for i in range(min(len(target), len(emitted_text)), 0, -1):
            if emitted_text.endswith(target[:i]):
                return target[:i]
        return ""

    def get_valid_token_ids(self, emitted_text: str) -> List[int]:
        """
        The Hot Loop Router: Returns allowed tokens instantly based on state.
        """
        if self.state == State.EXPECT_OPEN_BRACE:
            return self.cache_open_brace

        elif self.state == State.EXPECT_NAME_KEY:
            emitted_chunk = self._get_already_emitted('"name":', emitted_text)
            return self.cache_name_key.get(emitted_chunk, [])

        elif self.state == State.EXPECT_PARAMS_KEY:
            emitted_chunk = self._get_already_emitted(',"parameters":{', emitted_text)
            return self.cache_params_key.get(emitted_chunk, [])

        return []
