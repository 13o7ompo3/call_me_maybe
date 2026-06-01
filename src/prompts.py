from typing import List
from src.schemas import FunctionDefinition


def build_routing_prompt(user_query: str, functions: List[FunctionDefinition]
                         ) -> str:
    """
    Builds a hyper-dense, token-optimized prompt.
    Forces the LLM to act as a strict routing engine, abandoning chat behavior.
    """
    prompt = (
        "ROUTING ENGINE MODE ACTIVE.\n"
        "Read the user query, extract 3 keywords to understand the intent,\
         then output the function name.\n"
        "Do not explain. Use the exact format: \
        keyword1, keyword2, keyword3 | fn_name\n\n"
        "AVAILABLE FUNCTIONS:\n"
    )

    for f in functions:
        prompt += f"- {f.name}: {f.description}\n, returns {f.returns}\n"

    prompt += "\nEXAMPLES:\n"
    prompt += "Query: 'Say hello to Alice'\n\
ANALYSIS: greeting, welcome, person | fn_function\n"

    prompt += f"\nUSER QUERY: {user_query}\n"

    prompt += "ANALYSIS: "

    return prompt
