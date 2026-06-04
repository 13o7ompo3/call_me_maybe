 *This project has been created as part of the 42 curriculum by obahya.*


## Description

This project is a lightweight Function Calling and Parameter Extraction Engine optimized for Micro-LLMs (< 1 Billion parameters). It translates natural language user queries into structured, executable JSON function calls.


## Algorithm Explanation

The engine uses Constrained Decoding and Prefix Injection:

1. **Routing:** Maps the user query to the correct function using contextual logit masking (constrained decoding).

2. **Schema-Driven Tag Forcing:** The Python script forcefully injects XML start tags directly into the LLM's token buffer, forcing it to generate specific parameter values.

3. **Lookahead Prefix Matching:** The engine pre-computes targets (quoted strings, pure numbers, constants) from the prompt. As the LLM generates tokens, a buffer tracks the output. If a match is detected, the Python script hijacks the generation, auto-completes the value, and injects the closing tag.


## Design Decisions

* **XML over JSON:** XML is used during the generation phase because tags act as explicit state-transition markers, preventing the parsing errors common with incomplete JSON.

* **Boot-Time Parameter Grounding:** Aggressive, hardcoded hints are injected at runtime to guide the LLM and bypass type ambiguity without altering the core schema.

* **O(1) Token Extension:** Token arrays are updated using `current_ids.extend()` instead of re-encoding the entire sequence, eliminating $O(N^2)$ bottlenecks.

* **Deep Copying:** `copy.deepcopy()` is used to ensure state target dictionaries remain isolated and pristine across multiple parameter loops.


## Performance Analysis

* **Accuracy:** Effectively handles complex edge cases, including double-escaped regex (`\\d+`) and strict quote matching, by delegating extraction to Python regex when applicable.

* **Speed:** Lookahead prefix matching frequently cuts off the LLM after 1-2 tokens, drastically reducing inference latency.

* **Reliability:** By manually injecting schema tags, the engine guarantees zero missing keys and perfect structural integrity in the final JSON.


## Challenges Faced

* **Scope Leaks:** Python's shallow copy behavior caused dictionaries to mutate across loop iterations, leading to hallucinated values. Resolved by implementing deep copies.


## Testing Strategy



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



