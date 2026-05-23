import numpy as np
from typing import Dict, Any, Tuple
from llm_sdk import Small_LLM_Model
from src.vocab import Vocabulary
from src.schemas import FunctionDefinition
import re


class ExtractionGenerator:
    def __init__(self, llm: Small_LLM_Model, vocab: Vocabulary,
                 hints: Dict[str, Dict[str, str]]):
        self.llm = llm
        self.vocab = vocab
        self.hints = hints

    def _build_prompt(self, user_query: str, func_name: str,
                      func_def: FunctionDefinition) -> str:
        prompt = (
            "EXTRACTION ENGINE MODE ACTIVE.\n"
            f"Extract the exact arguments for '{func_name}' function "
            "from the user query.\nDO NOT SOLVE THE PROBLEM. \
            JUST COPY THE RAW arguments FROM THE QUERY.\n\n"
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

        first_tag = list(func_def.parameters.keys())[0] if func_def.parameters else ""
        if first_tag:
            prompt += f"<{first_tag}>"

        return prompt

    def extract(self, user_query: str, func_name: str,
                func_def: FunctionDefinition) -> Dict[str, Any]:
        if not func_def.parameters:
            return {}

        prompt = self._build_prompt(user_query, func_name, func_def)
        input_tensor = self.llm.encode(prompt)
        current_ids = input_tensor.tolist()[0]

        keys = list(func_def.parameters.keys())
        current_index = 0
        argument = ""
        self.probable_argument = self.get_probable_argument(user_query)
        probable_argument = self.probable_argument.copy()

        # Initialize with our pre-filled tag
        generated_text = f"<{keys[0]}>"
        print(f"\n[EXTRACTION] {generated_text}", end="", flush=True)

        # Standard greedy decoding loop
        for step in range(100):
            logits = self.llm.get_logits_from_input_ids(current_ids)
            next_id = int(np.argmax(logits))
            clean_str = self.vocab.id_to_token[next_id].replace(
                "Ġ", " ").replace("\u0120", " ").replace("Ċ", "\n")

            current_ids.append(next_id)
            generated_text += clean_str.lstrip()
            argument += clean_str
            print(clean_str, end="", flush=True)

            stripped_token = clean_str.lstrip()
            for level in probable_argument:
                for p_name, p_remaining in level.copy().items():
                    if p_remaining.startswith(stripped_token):
                        level[p_name] = p_remaining[len(stripped_token):]
                    else:
                        del level[p_name]
                if len(level) == 1:
                    # 1. Get the actual surviving key and value (Ignore the leaked loop variables!)
                    surviving_key = list(level.keys())[0]
                    surviving_remainder = level[surviving_key]
    
                    # 2. Use the REAL XML tag from your schema, not the dictionary key!
                    real_xml_tag = keys[current_index]
    
                    # 3. Build the correct prediction
                    prediction = f"{surviving_remainder}</{real_xml_tag}>\n"

                    tokens = self.llm.encode(prediction).tolist()[0]
                    current_ids.extend(tokens)
                    generated_text += prediction
                    print(prediction, end="", flush=True)
                if len(level) == 0:
                    probable_argument.remove(level)

            target_end_tag = f"</{keys[current_index]}>"

            if generated_text.strip().endswith(target_end_tag):
                current_index += 1
                argument = ""
                probable_argument = self.get_probable_argument(user_query)

                if current_index == len(keys):
                    print("Extraction complete!")
                    break

                next_tag = f"<{keys[current_index]}>"
                generated_text += next_tag

                new_tokens = self.llm.encode(next_tag).tolist()[0]
                current_ids.extend(new_tokens)
                print(next_tag, end="", flush=True)

        print("\n[EXTRACTION COMPLETE]")
        return self._parse_xml(generated_text, func_def)

    def _parse_xml(self, xml_string: str, func_def: FunctionDefinition
                   ) -> Dict[str, Any]:
        """Safely slices the XML string into a clean Python Dictionary."""
        result: Dict[str, Any] = {}
        for p_name, p_data in func_def.parameters.items():
            start_tag = f"<{p_name}>"
            end_tag = f"</{p_name}>"

            s_idx = xml_string.find(start_tag)
            e_idx = xml_string.find(end_tag)

            if s_idx != -1 and e_idx != -1:
                val_str = xml_string[s_idx + len(start_tag):e_idx].strip()

                if p_data.type in ("number", "integer"):
                    try:
                        result[p_name] = float(val_str) if p_data.type == "number" else int(val_str)
                    except ValueError:
                        result[p_name] = 0
                else:
                    result[p_name] = val_str
            else:
                result[p_name] = 0 if p_data.type in ("number", "integer") else ""

        return result

    def get_probable_argument(self, user_query: str):
        ret = []
        if ":" in user_query:
            parts = user_query.split(":")
            ret.append({part.strip(): part.strip() for part in parts})
        quoted_targets = re.findall(r'["\'](.*?)["\']', user_query)
        if quoted_targets:
            ret.append({qt.strip(): qt.strip() for qt in quoted_targets})
        ret.append({word: word for word in user_query.split()})
        return ret
