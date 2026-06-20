from typing import List
from src.schemas import FunctionDefinition


def build_routing_prompt(user_query: str, functions: List[FunctionDefinition]
                         ) -> str:
    """Build the prompt used to route a query to a single function.

    Args:
        user_query (str): Natural-language query from the user.
        functions (List[FunctionDefinition]): Candidate functions available for
            routing.

    Returns:
        str: Prompt instructing the model to emit a function name after a pipe.

    Raises:
        None.
    """
    prompt = (
        "ROUTING ENGINE MODE ACTIVE.\n"
        "Read the user query to understand the intent"
        " then output the function name.\n"
        "Do not explain. Use the exact format: function_name\n\n"
        "AVAILABLE FUNCTIONS:\n"
    )

    for f in functions:
        prompt += f"- {f.name}: {f.description}\n, returns {f.returns}\n"

    prompt += "\nEXAMPLES:\n"
    prompt += "Query: 'Say hello to Alice'\nFunction: fn_function\n"

    prompt += f"\nUSER QUERY: {user_query}\n"

    prompt += "Function: "

    return prompt
