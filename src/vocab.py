import json
import sys
from llm_sdk import Small_LLM_Model


class Vocabulary:
    def __init__(self, llm_model: str | None = None) -> None:
        print("Booting the LLM Engine...")
        if llm_model is not None:
            self.llm = Small_LLM_Model(model_name=llm_model)
        else:
            self.llm = Small_LLM_Model()

        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

        self._load()

    def _load(self) -> None:
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
