import numpy as np
import copy
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from os.path import commonprefix
from typing import Dict, Any
from src.vocab import Vocabulary
from src.schemas import FunctionDefinition
from pydantic import BaseModel, ConfigDict
import re


class ExtractionGenerator(BaseModel):
    """Extract function arguments from a user query with constrained decoding.

    Args:
        llm (Small_LLM_Model): Language model wrapper used for tokenization and
            logits.
        vocab (Vocabulary): Vocabulary helper that maps token IDs to strings.
        hints (Dict[str, Dict[str, str]]): Optional per-function extraction
            hints keyed by function and parameter name.

    Returns:
        None.

    Raises:
        None.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    llm: Small_LLM_Model
    vocab: Vocabulary
    hints: Dict[str, Dict[str, str]]
    probable_argument: list[Dict[str, str]] = []

    def _build_prompt(self, user_query: str, func_name: str,
                      func_def: FunctionDefinition) -> str:
        """Build the prompt used to extract arguments for one function.

        Args:
            user_query (str): Natural-language query from the user.
            func_name (str): Name of the function being extracted.
            func_def (FunctionDefinition): Function schema that defines
                the expected parameters and return type.

        Returns:
            str: Prompt instructing the model to emit XML-like argument tags.

        Raises:
            None.
        """
        prompt = (
            "EXTRACTION ENGINE MODE ACTIVE.\n"
            f"Extract the exact arguments for '{func_name}' function "
            "from the user query.\nDO NOT SOLVE THE PROBLEM. "
            "JUST COPY THE RAW arguments FROM THE QUERY.\n\n"
        )
        prompt += f"Function: {func_name}\n"
        prompt += f"Description: {func_def.description}\n"
        prompt += f"Returns: {func_def.returns}\n"
        prompt += "PARAMETERS TO EXTRACT:\n"
        for i, (p_name, p_data) in enumerate(func_def.parameters.items()):
            prompt += f"- name: {p_name}, type: {p_data.type}\n"
            if self.hints.get(func_name, {}).get(p_name):
                prompt += f"  hint: {self.hints[func_name][p_name]}\n"

        prompt += "\nEXAMPLES:\n"
        prompt += "Query: 'What is the square root of 81?'\n<a>81</a>\n"
        prompt += "Query: 'Greet Alice'\n<name>Alice</name>\n"

        prompt += f"\nUSER QUERY: {user_query}\nOUTPUT FORMAT:\n"
        for p_name in func_def.parameters.keys():
            prompt += f"<{p_name}></{p_name}>\n"

        prompt += "\nVALUES:\n"

        first_tag = list(func_def.parameters.keys())[0]
        if first_tag:
            prompt += f"<{first_tag}>"

        return prompt

    def extract(self, user_query: str, func_name: str,
                func_def: FunctionDefinition) -> Dict[str, Any]:
        """Extract typed arguments for a selected function.

        Args:
            user_query (str): Natural-language query from the user.
            func_name (str): Name of the function to extract arguments for.
            func_def (FunctionDefinition): Function schema describing
                the parameter names and types.

        Returns:
            Dict[str, Any]: Parsed arguments keyed by parameter name.

        Raises:
            None.
        """
        if not func_def.parameters:
            return {}

        prompt = self._build_prompt(user_query, func_name, func_def)
        input_tensor = self.llm.encode(prompt)
        current_ids = input_tensor.tolist()[0]

        keys = list(func_def.parameters.keys())
        current_index = 0
        argument = ""
        self.probable_argument = self.get_probable_argument(user_query)
        probable_argument = copy.deepcopy(self.probable_argument)

        # Initialize with our pre-filled tag
        generated_text = f"<{keys[0]}>"
        print(f"[EXTRACTION]\n\n{generated_text}", end="", flush=True)

        while True:  # Not safe anymore
            logits = self.llm.get_logits_from_input_ids(current_ids)
            next_id = int(np.argmax(logits))
            clean_str = self.vocab.id_to_token[next_id].replace(
                "Ġ", " ").replace("Ċ", "\n")

            current_ids.append(next_id)
            generated_text += clean_str
            argument += clean_str
            print(clean_str, end="", flush=True)

            stripped_token = clean_str.lstrip()
            for level in probable_argument:
                for p_name, p_remaining in level.copy().items():
                    if p_remaining.lstrip().startswith(stripped_token):
                        level[p_name] = p_remaining[len(stripped_token):]
                    else:
                        del level[p_name]
                if len(level) == 1:
                    surviving_key = list(level.keys())[0]
                    surviving_remainder = level[surviving_key]

                    real_xml_tag = keys[current_index]

                    prediction = f"{surviving_remainder}</{real_xml_tag}>\n"

                    tokens = self.llm.encode(prediction).tolist()[0]
                    current_ids.extend(tokens)
                    generated_text += prediction
                    print(prediction, end="", flush=True)
                    probable_argument.remove(level)
                    break
                else:
                    for p_name, p_remaining in level.items():
                        level[p_name] = p_remaining.lstrip()
                if len(level) == 0:
                    probable_argument.remove(level)
                prefix = commonprefix(list(level.values()))
                if prefix:
                    tokens = self.llm.encode(prefix).tolist()[0]
                    current_ids.extend(tokens)
                    generated_text += prefix
                    print(prefix, end="", flush=True)
                    for p_name, p_remaining in level.items():
                        level[p_name] = p_remaining[len(prefix):]
                    probable_argument = [level]
                    break

            target_end_tag = f"</{keys[current_index]}>"

            if generated_text.strip().endswith(target_end_tag):
                current_index += 1
                argument = ""
                probable_argument = copy.deepcopy(self.probable_argument)

                if current_index == len(keys):
                    print("\nExtraction complete!")
                    break

                next_tag = f"<{keys[current_index]}>"
                generated_text += next_tag

                new_tokens = self.llm.encode(next_tag).tolist()[0]
                current_ids.extend(new_tokens)
                print(next_tag, end="", flush=True)

        return self._parse_xml(generated_text, func_def)

    def _parse_xml(self, xml_string: str, func_def: FunctionDefinition
                   ) -> Dict[str, Any]:
        """Parse the generated XML-like text into a typed parameter mapping.

        Args:
            xml_string (str): Generated XML-like text containing parameters.
            func_def (FunctionDefinition): Function schema used to
                coerce types.

        Returns:
            Dict[str, Any]: Parsed parameter values with types coerced when
            possible.

        Raises:
            None.
        """
        result: Dict[str, Any] = {}
        for p_name, p_data in func_def.parameters.items():
            start_tag = f"<{p_name}>"
            end_tag = f"</{p_name}>"

            s_idx = xml_string.find(start_tag)
            e_idx = xml_string.find(end_tag)

            f = {"number": float, "integer": int}.get(p_data.type, str)
            if s_idx != -1 and e_idx != -1:
                val_str = xml_string[s_idx + len(start_tag):e_idx].strip()

                if p_data.type in ("number", "integer"):
                    try:
                        result[p_name] = f(val_str)
                    except ValueError:
                        result[p_name] = f(0)
                else:
                    result[p_name] = val_str
            else:
                if p_data.type in ("number", "integer"):
                    result[p_name] = f(0)
                else:
                    result[p_name] = ""

        return result

    def get_probable_argument(self, user_query: str) -> list[Dict[str, str]]:
        """Build candidate argument substrings from the user query.

        Args:
            user_query (str): Natural-language query that may contain
                likely argument values.

        Returns:
            list[Dict[str, str]]: Ordered candidate maps used to bias
            argument extraction.

        Raises:
            None.
        """
        ret = []
        if ":" in user_query:
            parts = user_query.split(":")
            ret.append({part.strip(): part.strip() for part in parts})
        quoted_targets = re.findall(r'(["\'])(.*?)\1', user_query)
        if quoted_targets:
            ret.append({qt[1].strip(): qt[1].strip() for qt in quoted_targets})
        clean_numbers = re.findall(r'\b\d+(?:\.\d+)?\b', user_query)
        if clean_numbers:
            ret.append({num: num for num in clean_numbers})
        ret.append({word: word for word in user_query.split()})
        return ret
