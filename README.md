 *This project has been created as part of the 42 curriculum by obahya.*


## Description

This project is a lightweight Function Calling and Parameter Extraction Engine optimized for Micro-LLMs (< 1 Billion parameters). It translates natural language user queries into structured, executable JSON function calls.


## Architecture & Algorithms

* **Routing via Relative Probability**: Maps queries to functions by isolating allowed token logits and applying Softmax strictly over the allowed subset. If the highest allowed token probability falls below 0.6, the engine aborts and defaults to fn_unsupported_action.

* **Custom BPE Tokenizer**: A native Python implementation that handles BPE whitespace artifacts (Ġ, Ċ) directly within the routing loop.

* **Schema-Driven Tag Forcing**: Injects XML start tags directly into the token buffer to enforce specific parameter generation.

## Design Decisions

* **XML over JSON:** XML is used during the generation phase because tags act as explicit state-transition markers, preventing the parsing errors common with incomplete JSON.

* **Boot-Time Parameter Grounding:** Aggressive, hardcoded hints are injected at runtime to guide the LLM and bypass type ambiguity without altering the core schema.


## Performance Analysis

* **Accuracy:** Relative probability thresholding and honeypots effectively eliminate false positives during function routing.

* **Speed:** Lookahead prefix matching frequently cuts off the LLM after 1-2 tokens, drastically reducing inference latency.

* **Reliability:** By manually injecting schema tags, the engine guarantees zero missing keys and perfect structural integrity in the final JSON.


## Challenges Faced

* **Scope Leaks:** Python's shallow copy behavior caused dictionaries to mutate across loop iterations, leading to hallucinated values. Resolved by implementing deep copies.


## Testing Strategy

Softmax & Routing Thresholds: Evaluates the constrained decoding math. Tests ensure that when the isolated Softmax probability of the best *allowed* token falls below the `0.6` relative threshold, the engine successfully aborts the generation loop and falls back to `fn_unsupported_action`.

## Instructions

**Prerequisites:**

Ensure you have the required LLM weights and Python dependencies installed.

```bash

uv sync

```


**Execution:**

Execute the program and run the test suite using the provided Makefile:

```bash

make run

```


## Example Usage

**Input Query:**

```json

{

"prompt": "Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat'"

}

```

**Execution Output:**

```

- Prompt: 'Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat''


ANALYSIS: cat, mat, cat |fn_sub


→ function: fn_substitute_string_with_regex


[EXTRACTION]


<source_string> The cat sat on the mat with another cat</source_string>

<regex>cat</regex>

<replacement>dog</replacement>


Extraction complete!


→ arguments: {'source_string': 'The cat sat on the mat with another cat', 'regex': 'cat', 'replacement': 'dog'}

```

**Final JSON:**

```json

{

"prompt": "Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat'",

"name": "fn_substitute_string_with_regex",

"parameters": {

"source_string": "The cat sat on the mat with another cat",

"regex": "cat",

"replacement": "dog"

}

}

```


## Resources

* **[Pydantic Documentation: Arbitrary Types & Validators](https://docs.pydantic.dev/latest/concepts/models/)** - For configuring `model_config` to accept custom LLM classes and managing post-initialization state.
* **[Andrej Karpathy's minbpe](https://github.com/karpathy/minbpe)** - The gold standard reference for understanding the pure Python implementation of Byte-Pair Encoding.
