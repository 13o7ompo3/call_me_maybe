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
        "You are a function calling system."
        "Read the user query and output "
        "EXACTLY AND ONLY the name of the correct function.\n"
        "\n\n"
    )

    prompt += "AVAILABLE FUNCTIONS:\n"
    for f in functions:
        prompt += f"- {f.name}: {f.description}\n, returns {f.return_type}\n"

    prompt += f"\nUSER QUERY: {user_query}\n"

    prompt += "SELECTED FUNCTION: fn_"

    return prompt
