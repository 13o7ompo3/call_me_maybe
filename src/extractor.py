import numpy as np
from typing import Dict, Any, Tuple
from llm_sdk import Small_LLM_Model
from src.vocab import Vocabulary
from src.schemas import FunctionDefinition


class ExtractionGenerator:
    def __init__(self, llm: Small_LLM_Model, vocab: Vocabulary):
        self.llm = llm
        self.vocab = vocab

    def _build_prompt(self, user_query: str, func_name: str,
                      func_def: FunctionDefinition) -> Tuple[str, str]:
        prompt = (
            "EXTRACTION ENGINE MODE ACTIVE.\n"
            f"Extract the exact values for '{func_name}' from the user query."
            "\nCRITICAL: DO NOT SOLVE THE PROBLEM. \
            JUST COPY THE RAW VALUES FROM THE TEXT.\n\n"
        )
        prompt += f"Function: {func_name}\n"
        prompt += f"Description: {func_def.description}\n"
        prompt += f"Returns: {func_def.returns}\n"
        prompt += "PARAMETERS TO EXTRACT:\n"
        for i, (p_name, p_data) in enumerate(func_def.parameters.items()):
            prompt += f"- name: {p_name}, type: {p_data.type}\n"

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

        return prompt, first_tag

    def extract(self, user_query: str, func_name: str,
                func_def: FunctionDefinition) -> Dict[str, Any]:
        if not func_def.parameters:
            return {}

        prompt, first_tag = self._build_prompt(user_query, func_name, func_def)
        input_tensor = self.llm.encode(prompt)
        current_ids = input_tensor.tolist()[0]

        # Initialize with our pre-filled tag
        generated_text = f"<{first_tag}>"
        print(f"\n[EXTRACTION] {generated_text}", end="", flush=True)

        # The target string that tells us to stop generating
        last_param = list(func_def.parameters.keys())[-1]
        end_marker = f"</{last_param}>"

        # Standard greedy decoding loop
        for step in range(100):
            logits = self.llm.get_logits_from_input_ids(current_ids)
            next_id = int(np.argmax(logits))
            clean_str = self.vocab.id_to_token[next_id].replace(
                "Ġ", " ").replace("\u0120", " ").replace("Ċ", "\n")

            current_ids.append(next_id)
            generated_text += clean_str
            print(clean_str, end="", flush=True)

            if end_marker in generated_text:
                break

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
                        result[p_name] = float(val_str) if "." in val_str else int(val_str)
                    except ValueError:
                        result[p_name] = 0
                else:
                    result[p_name] = val_str

        return result
