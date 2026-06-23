from typing import List, Dict
from pydantic import BaseModel, model_validator, Field


class RouterCache(BaseModel):
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

    vocab: Dict[int, str]
    allowed_functions: List[str]
    valid_prefixes: Dict[str, List[int]] = Field(default={})

    @model_validator(mode="after")
    def init(self) -> 'RouterCache':
        self.valid_prefixes = self._build_prefix_cache(self.allowed_functions)
        return self

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

                for j in range(1, len(remaining) + 1):
                    potential_token = remaining[:j]
                    if potential_token in clean_vocab:
                        cache[already_emitted].extend(
                            clean_vocab[potential_token])

        for k in cache:
            cache[k] = list(set(cache[k]))

        return cache

    def get_valid_token_ids(self, emitted_text: str) -> List[int]:
        """Return the allowed token IDs for the current emitted prefix.

        Args:
            emitted_text (str): Text already generated.

        Returns:
            List[int]: Valid token IDs for continuing the current prefix, or an
            empty list when the prefix is unknown.

        Raises:
            None.
        """
        return self.valid_prefixes.get(emitted_text, [])
