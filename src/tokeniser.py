from typing import List
from .vocab import Vocabulary
from pydantic import BaseModel, ConfigDict


class Tokeniser(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    vocab: Vocabulary

    def decode(self, tokens: List[int]) -> str:
        """Decode a list of token IDs into a string.
        Args:
            tokens (List[int]): A list of token IDs to decode.
        Returns:
            str: The decoded string corresponding to the input token IDs.
        Raises:
            ValueError: If a token ID is not found in the vocabulary.
        """
        return "".join(self.vocab.id_to_token[token].replace('Ġ', ' ')
                       .replace("Ċ", "\n").replace("ĉ", "\t")
                       for token in tokens if token in self.vocab.id_to_token)

    def encode(self, text: str) -> List[int]:
        """Encode a string into a list of token IDs using BPE tokenization.
        Args:
            text (str): The input string to encode.
        Returns:
            List[int]: A list of token IDs corresponding to the input string.
        Raises:
            ValueError: If a character in the input string is not found
                        in the vocabulary.
            """
        tokens = []
        text = text.replace(' ', 'Ġ').replace("\n", "Ċ").replace("\t", "ĉ")
        for char in text:
            if char in self.vocab.token_to_id:
                tokens.append(self.vocab.token_to_id[char])
            else:
                raise ValueError(f"Character '{char}' not found"
                                 " in vocabulary.")
        while len(tokens) > 1:
            pairs = set(zip(tokens[:-1], tokens[1:]))
            merge_candidates = []
            for pair in pairs:
                if pair in self.vocab.merge:
                    merge_candidates.append((pair, self.vocab.merge[pair]))

            if not merge_candidates:
                break

            best_pair, token_id_and_rank = min(merge_candidates,
                                               key=lambda x: x[1][1])
            merged_token_id = token_id_and_rank[0]
            new_tokens: List[int] = []
            i = 0
            while i < len(tokens):
                if (i < len(tokens) - 1
                   and (tokens[i], tokens[i + 1]) == best_pair):
                    new_tokens.append(merged_token_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens
