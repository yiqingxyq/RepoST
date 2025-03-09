import os 
docker_CACHE_DIR = os.environ.get("docker_CACHE_DIR")

sandbox_prompt_template = """\
Instructions:
- You're given a piece of PYTHON CODE containing a function called {func_name}. Your goal is to revise the PYTHON CODE so that we can directly call the {func_name} function WITHOUT ANY MODIFICATIONS.
- You should edit the original PYTHON CODE as little as possible and you can add code only if necessary.
- DO NOT call any external API, database, etc. Instead, create a mock interface.
- Make sure that your code can be directly executed without any modification. For example, statements like `token = "your_auth_token_here"  # You need to replace this with a real token` is NOT allowed.
- If you need to read files, write example files to the `{docker_CACHE_DIR}` directory.

- Provide your reasoning and the revised PYTHON CODE below SOLUTION.

PYTHON CODE:
```python
{code}
```

Your answer should follow the format below:

Reasoning: ...
```python
# Your Code. 
```

Do NOT include other formatting. Output every token of the content with no omission or abbreviation.

SOLUTION:
"""

aggregate_and_sandbox_prompt_template = """\
Instructions:
- You're given a piece of PYTHON CODE containing a function called {func_name}. We also provide you the CONTEXT of the PYTHON CODE. Your goal is to aggregate the PYTHON CODE and the CONTEXT into one script, so that we can directly call the {func_name} function WITHOUT ANY MODIFICATIONS.
- You should edit the original PYTHON CODE as little as possible and you can add code only if necessary.
- DO NOT call any external API, database, etc. Instead, create a mock interface.
- Make sure that your code can be directly executed without any modification. For example, statements like `token = "your_auth_token_here"  # You need to replace this with a real token` is NOT allowed.
- If you need to write files to the disk, use `{docker_CACHE_DIR}` as the directory.

- Provide your reasoning and the revised PYTHON CODE below SOLUTION.

PYTHON CODE:
```python
{code}
```

CONTEXT:
{context}

Your answer should follow the format below:

Reasoning: ...
```python
# Your Code. 
```

Do NOT include other formatting. Output every token of the content with no omission or abbreviation. For example, abbreviation like `... # the code keeps unchanged` is NOT allowed.

SOLUTION:
"""

test_generation_prompt_template = """\
Instructions:
- You're given a piece of PYTHON CODE containing a function called {func_name}. Assume we will later have another implentation of the {func_name} function called {func_name}_new_implementation.
- Your goal is to add (1) a test function called {test_func_name} to check whether {func_name}_new_implementation has the same functionality as the {func_name} function, and (2) a __main__ function that calls the test function.
- If the PYTHON CODE already contains a __main__ function, remove it and write a new __main__ function.
- The test function {test_func_name} should contain at least 3 assert statements. If {func_name}_new_implementation has different functionality as {func_name}, an Assertion Error should be triggered.
- The test function {test_func_name} should cover all the major branches of the {func_name} function
- DO NOT test on error handling and DO NOT test on the print information in the function.
- The __main__ function should NOT contain a try-except structure. If the implementation is incorrect, the program should have a non-zero exit code.
- You should edit the original PYTHON CODE as little as possible.
- If you need to write files to the disk, use `{docker_CACHE_DIR}` as the directory.

- Provide your reasoning and the new PYTHON CODE containing your test function {test_func_name} and the __main__ function below SOLUTION.

PYTHON CODE:
```python
{code}
```

Your answer should follow the format below:

Reasoning: ...
```python
# The new PYTHON CODE containing your test function {test_func_name} and the __main__ function.
```

Do NOT include other formatting. Output every token of your edited PYTHON CODE with no omission or abbreviation.

SOLUTION:
"""

debug_prompt_template = """\
Instructions:
- You're given a piece of PYTHON CODE containing a function called {func_name} and its test function called {test_func_name}. Assume we will later add another function called {func_name}_new_implementation, the test function aims to check whether {func_name}_new_implementation has the same functionality as {func_name}.
- In our experiments, we implemented {func_name}_new_implementation exactly the same as {func_name}, but the PYTHON CODE cannot be successfully executed. 
- Your task is to debug PYTHON CODE based on the ERROR MESSAGE.
- You should modify the code as little as possible, especially the test_{func_name} function and the {func_name} function.
- Make sure that after debugging, the test function test_{func_name} still have at least three assert statements and cover all the major branches of the {func_name} function.
- DO NOT test the logging information of error handling and DO NOT test on the print information in the function.
- If you need to write files to the disk, use `{docker_CACHE_DIR}` as the directory.

- Provide your reasoning and the debugged PYTHON CODE below SOLUTION. If necessary, output the bash scripts for Linux in another code block in the format of ```bash ... ```.

PYTHON CODE:
```python
{code}
```

ERROR MESSAGE:
```
{err_msg}
```

Your answer should follow the format below:

Reasoning: ...
```python
# The debugged PYTHON CODE in one piece.
```
```bash 
# the bash script, if necessary
```

Do NOT include other formatting. Output every token of your debugged PYTHON CODE with no omission or abbreviation.

SOLUTION:
"""

coverage_prompt_template = """\
Instructions:
- You're given a piece of PYTHON CODE containing a function called {func_name} and its test function called {test_func_name}. Assume we will later add another function called {func_name}_new_implementation, the test function aims to check whether {func_name}_new_implementation has the same functionality as {func_name}.
- You're also given the MISSING LINES of the {func_name}_new_implementation function that are NOT covered by {test_func_name}.
- Your task is to improve the branch coverage rate of the {test_func_name} function.
- You should only modify the {test_func_name} function. DO NOT modify other parts of the code.
- DO NOT test the logging information of error handling and DO NOT test on the print information in the function.
- If you need to write files to the disk, use `{docker_CACHE_DIR}` as the directory.

- Provide your reasoning and your revised {test_func_name} function below SOLUTION.

PYTHON CODE:
```python
{code}
```

MISSING LINES:
{missing_code}

Your answer should follow the format below:

Reasoning: ...
```python
# Your revised {test_func_name} function
```

Do NOT include other formatting. Output every token of the {test_func_name} function with no omission or abbreviation.

SOLUTION:
"""

more_tests_prompt_template = """\
Instructions:
- You're given a piece of PYTHON CODE containing a function called {func_name} and its test function called {test_func_name}. Assume we will later add another function called {func_name}_new_implementation, the test function aims to check whether {func_name}_new_implementation has the same functionality as {func_name}.
- Your task is to generate more tests for the {test_func_name} function.
- You should only modify the {test_func_name} function. DO NOT modify other parts of the code.
- DO NOT test the logging information of error handling and DO NOT test on the print information in the function.
- If you need to write files to the disk, use `{docker_CACHE_DIR}` as the directory.

- Provide your reasoning and your revised {test_func_name} function below SOLUTION.

PYTHON CODE:
```python
{code}
```

Your answer should follow the format below:

Reasoning: ...
```python
# Your revised {test_func_name} function
```

Do NOT include other formatting. Output every token of the {test_func_name} function with no omission or abbreviation.

SOLUTION:
"""


sandbox_check_prompt_template = """\
Instructions:
- We revised a python function called {func_name} so it can be directly executed in an isolated environment.
- You are given the ORIGINAL FUNCTION and the CODE containing the REVISED FUNCTION.
- Your task is to check whether the functionality of the REVISED FUNCTION is the same as the ORIGINAL FUNCTION.
- If the REVISED FUNCTION is exactly the same as the ORINIGAL FUNCTION, output "same" as your answer.
- Otherwise, if the functionality of the REVISED FUNCTION is the same as the ORIGINAL FUNCTION, output "yes" as your answer.
- if the functionality of the REVISED FUNCTION is different, output "no".

- Provide your reasoning and the answer under "SOLUTION".

ORIGINAL FUNCTION:
{orig_func}

CODE containing the REVISED FUNCTION:
{new_code}

Your answer should follow the format below:
```
REASONING: Your reasoning,
ANSWER: "same", "yes" or "no".
```

Do NOT include other formatting.

SOLUTION:
"""



test_check_prompt_template = """\
Instructions:
- You are given a piece of PYTHON CODE containing a function called {func_name}, its new implementation {func_name}_new_implementation (now hidden) and its test function called {test_func_name}.
- Your task is to judge whether the test function satisfies all the CONDITIONS:
[CONDITION 1] The {func_name} function should either have return values or modifies global variables or input arguments (such as a list, a dictionary, a class, etc.).
[CONDITION 2] The test cases should only check the return values or variable states. It should NOT check printed or logged contents.
[CONDITION 3] {func_name}_new_implementation can pass all the test cases IF AND ONLY IF it has the EXACTLY same functionality as {func_name}.
[CONDITION 4] The test cases and assert statements are reasonable. For example, if {func_name} does not have return values, you should NOT use `assert {func_name}() == {func_name}_new_implementation()` to test the implementation.
[CONDITION 5] The test cases are non-trivial.

- If the test function satisfies all the CONDITIONS, answer "yes". Otherwise, answer "no".
- Provide your reasoning and the answer under "SOLUTION".

PYTHON CODE:
{code}

Your answer should follow the format below:
```
REASONING: Your reasoning,
ANSWER: "yes" or "no".
```

Do NOT include other formatting.

SOLUTION:
"""


inst_prompt = """\
Below is a piece of Python Code. Generate the docstring for the `{func_name}` function based on the context.

Python Code:
```
{code}
```

Format your answer in the following format:
\"\"\"
(your generated docstring)
\"\"\"
"""