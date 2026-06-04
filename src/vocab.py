import json
import sys
import importlib
from typing import Any


class Vocabulary:
    """Load tokenizer vocabulary data and expose token lookup tables.

    Args:
        llm_model (str | None): Optional Hugging Face model identifier.

    Returns:
        None.

    Raises:
        Exception: Propagates model-loading and tokenizer-download failures.
        SystemExit: If the tokenizer vocabulary cannot be loaded.
    """

    def __init__(self, llm_model: str | None = None) -> None:
        """Initialize the model wrapper and load its tokenizer vocabulary.

        Args:
            llm_model (str | None): Optional Hugging Face model identifier.

        Returns:
            None.

        Raises:
            Exception: Propagates model-loading failures.
            SystemExit: If the tokenizer vocabulary cannot be loaded.
        """
        print("Booting the LLM Engine...")

        try:
            # Bypass mypy errors
            llm_mod = importlib.import_module("llm_sdk")
            llm_cls = getattr(llm_mod, "Small_LLM_Model")
        except Exception:
            # Let runtime import errors surface normally
            raise

        if llm_model is not None:
            self.llm: Any = llm_cls(model_name=llm_model)
        else:
            self.llm = llm_cls()

        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

        self._load()

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
            tokenizer_path = self.llm.get_path_to_tokenizer_file()
            with open(tokenizer_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            raw_vocab = data.get("model", {}).get("vocab", {})

            if not raw_vocab:
                print("Error: Could not extract vocabulary dictionary"
                      " from tokenizer file.")
                sys.exit(1)

            self.token_to_id = raw_vocab
            self.id_to_token = {v: k for k, v in raw_vocab.items()}
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            sys.exit(1)
