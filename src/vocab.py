import json
import sys
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from pydantic import BaseModel, model_validator, ConfigDict


class Vocabulary(BaseModel):
    """Load tokenizer vocabulary data and expose token lookup tables.

    Args:
        None.
    Returns:
        None.

    Raises:
        Exception: Propagates model-loading and tokenizer-download failures.
        SystemExit: If the tokenizer vocabulary cannot be loaded.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    llm: Small_LLM_Model | None = None
    token_to_id: dict[str, int] = {}
    id_to_token: dict[int, str] = {}
    merge: dict[tuple[int, int], tuple[int, int]] = {}

    @model_validator(mode="after")
    def init(self) -> 'Vocabulary':
        if self.llm is None:
            self.llm = Small_LLM_Model()
            self.token_to_id: dict[str, int] = {}
            self.id_to_token: dict[int, str] = {}
            # {(token_id_1, token_id_2): (merged_token_id, merge_rank)}
            self.merge: dict[tuple[int, int], tuple[int, int]] = {}
            self._load()
        return self

    def _load(self) -> None:
        """Load tokenizer vocabulary data from the model repository.

        Args:
            None.

        Returns:
            None.

        Raises:
            SystemExit: If the tokenizer file is missing or malformed.
            Exception: Propagates unexpected download or file errors.
        """
        try:
            if self.llm is None:
                print("Error: LLM model is not initialized.")
                sys.exit(1)
            vocab_path = self.llm.get_path_to_vocab_file()
            with open(vocab_path, "r", encoding="utf-8") as f:
                raw_vocab = json.load(f)

            if not isinstance(raw_vocab, dict):
                print("Error: Vocabulary file is not a valid JSON object.")
                sys.exit(1)
            if not all(isinstance(k, str) and isinstance(v, int)
                       for k, v in raw_vocab.items()):
                print("Error: Invalid token_to_id mapping in vocabulary.")
                sys.exit(1)
            self.token_to_id = raw_vocab
            self.id_to_token = {v: k for k, v in raw_vocab.items()}

            merge_path = self.llm.get_path_to_merges_file()
            with open(merge_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    line = line.strip().split("#")[0]
                    if not line:
                        continue
                    try:
                        token1, token2 = line.strip().split()
                    except ValueError:
                        print(f"Error: Invalid merge line '{line}' "
                              "in merges file.")
                        sys.exit(1)
                    if (token1 not in self.token_to_id
                       or token2 not in self.token_to_id
                       or (token1 + token2) not in self.token_to_id):
                        print(f"Error: tokens in line '{line}'"
                              " are not in the vocabulary.")
                        sys.exit(1)
                    token_id1 = self.token_to_id[token1]
                    token_id2 = self.token_to_id[token2]
                    merged_token = token1 + token2
                    merged_token_id = self.token_to_id[merged_token]
                    self.merge[(token_id1, token_id2)] = (merged_token_id, i)
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            sys.exit(1)
