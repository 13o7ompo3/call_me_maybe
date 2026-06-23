import json
import sys
from llm_sdk.llm_sdk import Small_LLM_Model
from pydantic import BaseModel, model_validator


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

    @model_validator(mode="after")
    def init(self) -> 'Vocabulary':
        self.llm = Small_LLM_Model()
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
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
            vocab_path = self.llm.get_path_to_vocab_file()
            with open(vocab_path, "r", encoding="utf-8") as f:
                raw_vocab = json.load(f)

            self.token_to_id = raw_vocab
            self.id_to_token = {v: k for k, v in raw_vocab.items()}
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            sys.exit(1)
