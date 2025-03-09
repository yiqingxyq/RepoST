
debug_prompt = """\
Debug the BUGGY IMPLEMENTATION of the `{func_name}` function based on the CONTEXT and ERROR MESSAGE.

CONTEXT:
```
{context}
```

BUGGY IMPLEMENTATION:
```
{buggy_solution}
```

ERROR MESSAGE:
```
{error_msg}
```

NEW IMPLEMENTATION:
"""

debug_target = """\
```    
{solution}
```
"""

debug_prompt = """\
Debug the BUGGY IMPLEMENTATION of the `{func_name}` function based on the CONTEXT and ERROR MESSAGE.

Provide your reasoning and the debugged function body below NEW IMPLEMENTATION.

CONTEXT:
```
{context}
```

BUGGY IMPLEMENTATION:
```
{buggy_solution}
```

ERROR MESSAGE:
```
{error_msg}
```

Your answer should follow the format below:
Reasoning: ...
```python
# Your Code. 
```

NEW IMPLEMENTATION:
"""

debug_reasoning_target = """\
Reasoning: {reasoning}
```    
{solution}
```
"""