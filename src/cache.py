from typing import List, Dict


class RouterCache:
    """Cache valid next-token IDs for allowed function-name prefixes.

    Args:
        vocab_map (Dict[int, str]): Mapping from token IDs to token
            strings.
        allowed_functions (List[str]): Function names that may be
            generated.

    Returns:
        None.

    Raises:
        None.
    """

    def __init__(self, vocab_map: Dict[int, str], allowed_functions: List[str]
                 ) -> None:
        """Build the prefix cache for the allowed function names.

        Args:
            vocab_map (Dict[int, str]): Mapping from token IDs to token
                strings.
            allowed_functions (List[str]): Function names that may be
                generated.

        Returns:
            None.

        Raises:
            None.
        """
        print("Initializing High-Speed Router Cache...")
        self.vocab = vocab_map
        self.allowed_functions = allowed_functions

        self.valid_prefixes = self._build_prefix_cache(allowed_functions)
        print(f"[SUCCESS] Router Cache locked for {len(allowed_functions)}"
              " functions.")

    def _build_prefix_cache(self, functions: List[str]
                            ) -> Dict[str, List[int]]:
        """Build a prefix-to-token-ID lookup table for function names.

        Args:
            functions (List[str]): Function names to index in the cache.

        Returns:
            Dict[str, List[int]]: Mapping from emitted prefix text to valid
            next-token IDs.

        Raises:
            None.
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

                potential_token_id = []
                for j in range(1, len(remaining) + 1):
                    potential_token = remaining[:j]
                    if potential_token in clean_vocab:
                        potential_token_id = clean_vocab[potential_token]
                if potential_token_id:
                    cache[already_emitted].extend(potential_token_id)

        for k in cache:
            cache[k] = list(set(cache[k]))

        return cache

    def get_valid_token_ids(self, emitted_text: str) -> List[int]:
        """Return the allowed token IDs for the current emitted prefix.

        Args:
            emitted_text (str): Text already generated after the pipe symbol.

        Returns:
            List[int]: Valid token IDs for continuing the current prefix, or an
            empty list when the prefix is unknown.

        Raises:
            None.
        """
        return self.valid_prefixes.get(emitted_text, [])
