import json
from llm_sdk import Small_LLM_Model

# Initialize the engine
model = Small_LLM_Model()

# Extract the tokenizer mapping
# tokenizer_path = model.get_path_to_tokenizer_file()
# with open(tokenizer_path, "r") as f:
#     tokenizer_data = json.load(f)
    # Extract the token ID to string mapping here

prompt = "What is the sum of 2 and 3?"

# Encode returns a 2D tensor: tensor([[ 892, 318, ... ]])
input_tensor = model.encode(prompt)

# Convert to 1D list for the logits function
# input_ids = input_tensor.tolist()[0]

# Get the raw probabilities for the NEXT token
# logits = model.get_logits_from_input_ids(input_ids)
output = model.decode(input_tensor)
print(f"Output: {output}")

# --- CONSTRAINED DECODING MASK GOES HERE ---
