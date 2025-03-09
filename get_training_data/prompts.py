evaluation_prompt_template = """\
Instructions:
- Goal: Your goal is to complete the __func_name__ function in the __file_name__ file.
- The function you generate should align with the FUNCTION DESCRIPTION and the CONTEXT retrieved from the same repository.
- Output the complete function with the function decorators (if any).

- Provide your reasoning and the implementation of the complete __func_name__ function below SOLUTION. Wrap your function in a code block: ``` ... ```.

FUNCTION DESCRIPTION:
__instruction__

CONTEXT:
__context__

You should generate the compelte function with decorators (if any). Your answer should follow the format below:

Reasoning: ...
```python
__masked_func__
```

SOLUTION:
"""