from typing import List, Dict


class RouterCache:
    def __init__(self, vocab_map: Dict[int, str], allowed_functions: List[str]
                 ) -> None:
        print("Initializing High-Speed Router Cache...")
        self.vocab = vocab_map
        self.allowed_functions = allowed_functions

        self.valid_prefixes = self._build_prefix_cache(allowed_functions)
        print(f"[SUCCESS] Router Cache locked for {len(allowed_functions)}"
              " functions.")

    def _build_prefix_cache(self, functions: List[str]
                            ) -> Dict[str, List[int]]:
        """
        Maps every possible partial string of a function name
        to its valid next tokens.
        Example: cache['fn_a'] = [ID for 'dd', ID for 'dd_num', ID for 'd']
        """
        clean_vocab: Dict[str, List[int]] = {}
        for tid, tok_str in self.vocab.items():
            clean = tok_str.replace("Ġ", " ").replace("\u0120", " ")
            if not clean:
                continue
            if clean not in clean_vocab:
                clean_vocab[clean] = []
            clean_vocab[clean].append(tid)

        cache: Dict[str, List[int]] = {}

        for func_name in functions:
            for i in range(len(func_name)):
                already_emitted = func_name[:i]
                remaining = func_name[i:]

                if already_emitted not in cache:
                    cache[already_emitted] = []

                for j in range(1, len(remaining) + 1):
                    potential_token = remaining[:j]
                    if potential_token in clean_vocab:
                        cache[already_emitted].extend(
                            clean_vocab[potential_token])

        for k in cache:
            cache[k] = list(set(cache[k]))

        return cache

    def get_valid_token_ids(self, emitted_text: str) -> List[int]:
        """
        The O(1) Hot Loop.
        Instantly returns the allowed tokens
        for whatever text has been generated so far.
        """
        return self.valid_prefixes.get(emitted_text, [])
