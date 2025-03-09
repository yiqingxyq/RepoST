import os
import re 
import ast
import tokenize
import textwrap
from io import StringIO
from copy import deepcopy
from lib2to3 import refactor

import openai

client = None

CODE_BLOCK_PATTERN = r"```(\w*)\n(.*?)\n```"
def extract_code(text: str, pattern: str = CODE_BLOCK_PATTERN):
    match = re.findall(pattern, text, flags=re.DOTALL)
    return match if match else []


def safe_parse(code):
    try:
        return ast.parse(code)
    except SyntaxError as e:
        pass
    
    # add an empty line
    try:
        return ast.parse(code+"\n")
    except SyntaxError as e:
        pass
    
    # remove the last line if emptry
    lines = code.split("\n")
    while not lines[-1].strip():
        lines = lines[:-1]
    try:
        clean_code = "\n".join(lines)
        return ast.parse(clean_code)
    except:
        pass
    
    # Attempt to remove the BOM before parsing
    try:
        clean_code = code.lstrip("\ufeff")
        return ast.parse(clean_code)
    except:
        pass
    
    # Attempt to convert and retry
    tool = refactor.RefactoringTool(refactor.get_fixers_from_package("lib2to3.fixes"))
    try:
        py3_code = str(tool.refactor_string(code, "<string>"))
        return ast.parse(py3_code)
    except:
        pass
    
    try:
        py3_code = str(tool.refactor_string(code+"\n", "<string>"))
        return ast.parse(py3_code)
    except:
        pass


def list_files(startpath):
    output_lines = []
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        output_lines.append('{}{}/'.format(indent + "- ", os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            output_lines.append('{}{}'.format(subindent+ "- ", f))
            
    return "\n".join(output_lines)


def count_code_branches(func_code):
    dedented_func_code = textwrap.dedent(func_code)
    tree = safe_parse(dedented_func_code)
    
    branch_count = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
            branch_count += 1
    return branch_count


def count_code_tokens(code):
    code_io = StringIO(code)
    tokens = list(tokenize.generate_tokens(code_io.readline))
    return len(tokens)


def get_func_indent(script, func_name):
    function_lines = script.split("\n")
    for lid,line in enumerate(function_lines):
        if f"def {func_name}" in line:
            clean_code = line.lstrip()
            indent = line.split(clean_code)[0]
            return indent
        
    return None


def get_func_body_rel_indent(script, func_name):
    func_indent = None
    script = textwrap.dedent(script)
    function_lines = script.split("\n")
    for lid,line in enumerate(function_lines):
        clean_code = line.lstrip()
        indent = line.split(clean_code)[0]
        if f"def {func_name}" in line:
            func_indent = indent 
        
        if func_indent is not None and len(indent) > len(func_indent):
            return indent

    return None


def get_docstring(func_code, func_name):
    """
    Extracts the docstring of a Python function from its code.

    Parameters:
        func_code (str): The source code of the Python function as a string.

    Returns:
        str: The docstring of the function if it exists, otherwise None.
    """
    # Get indent 
    orig_indent = get_func_indent(func_code, func_name)
    
    # Parse the input script to an AST
    dedented_func_code = textwrap.dedent(func_code)
    tree = safe_parse(dedented_func_code)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return ast.get_docstring(node)

    return None


def get_code_block_by_path(context_blocks, path):
    for block in context_blocks:
        lines = [x for x in block.split("\n") if x and x[0] == "#"]
        if not lines:
            return None 
        first_line = lines[0]
        if first_line:
            if path in first_line:
                return block 
    return None


def check_func_body_match(script, func_code):
    return "".join(func_code.split()) in "".join(script.split())


def check_func_ast_match(script, func_code, func_name, class_name=None):
    # return "".join(func_code.split()) in "".join(script.split())
    
    if "".join(func_code.split()) in "".join(script.split()):
        return True
    
    start_idx, end_idx = get_function_line_idx(script, func_name, class_name)
    sandboxed_func_code = "\n".join( script.split("\n")[start_idx : end_idx+1] )
    sandboxed_func_code = textwrap.dedent(sandboxed_func_code)
    
    orig_func_code = textwrap.dedent(func_code)
    
    try:
        # sandboxed_ast = safe_parse(sandboxed_func_code)
        # orig_ast = safe_parse(orig_func_code)
        sandboxed_ast = remove_comments( remove_comments_by_re(sandboxed_func_code), func_name, return_type="node" )
        sandboxed_func_ast = next(node for node in sandboxed_ast.body if isinstance(node, ast.FunctionDef))
        
        orig_ast = remove_comments( remove_comments_by_re(orig_func_code), func_name, return_type="node" )
        orig_func_ast = next(node for node in orig_ast.body if isinstance(node, ast.FunctionDef))
        
        if len(sandboxed_func_ast.body) != len(orig_func_ast.body):
            return False 
        
        for s_body, o_body in zip(sandboxed_func_ast.body, orig_func_ast.body):
            if ast.dump(s_body) != ast.dump(o_body):
                return False
        
        return True
    except:
        return False


def get_class_function_line_idx(script, func_name, class_name):
    """
    Finds the start and end line numbers of a specific class method in a Python script.
    """
    start_line, end_line = None, None
    try:
        tree = safe_parse(script)

        # Iterate over nodes in the AST
        for node in ast.walk(tree):
            # Find the specified class
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                # Search for the specified function in the class
                for sub_node in node.body:
                    if isinstance(sub_node, ast.FunctionDef) and sub_node.name == func_name:
                        if start_line is not None:
                            # error: multiple occurrence of the same function
                            print(f"Error: Multiple occurrence of function: {class_name}.{func_name}")
                            return None, None 
                        
                        # Return the start and end line numbers of the function
                        start_line = sub_node.lineno - 1
                        # Using end_lineno (introduced in Python 3.8+)
                        end_line = getattr(sub_node, "end_lineno", None) - 1
                    
    except Exception as e:
        print(f"Error occurred when search for funcion {class_name}.{func_name}: {e}")

    return start_line, end_line

def get_standalone_function_line_idx(script, func_name):
    """
    Finds the start and end line numbers of a standalone function in a Python script.
    """
    start_line, end_line = None, None
    try:
        tree = safe_parse(script)

        # Iterate over nodes in the AST to find standalone functions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                if start_line is not None:
                    # error: multiple occurrence of the same function
                    print(f"Error: Multiple occurrence of function: {func_name}")
                    return None, None 
                
                # Return the start and end line numbers of the function
                start_line = node.lineno - 1
                # Using end_lineno (introduced in Python 3.8+)
                end_line = getattr(node, 'end_lineno', None) - 1
            
    except Exception as e:
        print(f"Error occurred when search for funcion {func_name}: {e}")

    return start_line, end_line


def get_function_line_idx(script, func_name, class_name=None):
    if class_name is not None:
        start_line, end_line = get_class_function_line_idx(script, func_name, class_name)
    else:
        start_line, end_line = get_standalone_function_line_idx(script, func_name)
        
    if start_line is None:
        return None, None
        
    # sth like @static
    lines = script.split("\n")
    if start_line > 0 and len( lines[start_line-1].strip() ) > 0:
        start_line = start_line - 1
        
    return start_line, end_line
    
def get_function_parent_line_idx(script, func_start_line):
    lines = script.split("\n")
    func_content = lines[func_start_line].lstrip()
    func_indent = lines[func_start_line].split(func_content)[0]
    
    parent_line_idx_list = []
    cur_indent = func_indent
    idx_list = list(range(len(lines)))
    for idx, line in zip(idx_list[::-1], lines[::-1]):
        if not line.strip():
            continue 
        
        content = line.lstrip()
        line_indent = line.split(content)[0]        
        if len(line_indent) < len(func_indent):
            cur_indent = line_indent
            parent_line_idx_list.append(idx)
            
        if len(line_indent) == 0:
            break
            
    return parent_line_idx_list
    
    
def get_func_body_line_idx(func_code, func_name):
    """
    Get the start and end line indices of a function body (excluding the docstring).

    Args:
        func_code (str): The entire code containing the function.
        func_name (str): The name of the function to analyze.

    Returns:
        tuple: A tuple containing the start and end line indices of the function body.
    """
    dedented_func_code = textwrap.dedent(func_code)
    tree = safe_parse(dedented_func_code)
    
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            start_line = node.body[0].lineno - 1
            if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                start_line = node.body[1].lineno - 1

            end_line = getattr(node, 'end_lineno', None) - 1
            return start_line, end_line
            
    return None
    
        
def mask_func_body(script, func_name, docstring=None, add_pass=True):
    # Get indent 
    orig_indent = get_func_indent(script, func_name)
    
    # Parse the input script to an AST
    dedented_script = textwrap.dedent(script)
    tree = safe_parse(dedented_script)

    # Define a custom transformer to modify the function body
    class ReplaceBodyTransformer(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.name == func_name:
                # Replace the function body with an Ellipsis node
                node.body = []
                if docstring is not None:
                    node.body.append( ast.Expr(value=ast.Str(s=docstring)) )
                if add_pass:
                    node.body.append( ast.Pass() )
            return node

    # Transform the AST
    transformer = ReplaceBodyTransformer()
    modified_tree = transformer.visit(tree)
    modified_code = ast.unparse(modified_tree)

    # Re-indent the modified code to match the original script's structure
    indented_code = textwrap.indent(modified_code, orig_indent)
    return indented_code
    

def remove_comments(func_script, func_name, return_type="text"):
    """
    Removes the comments from a given function script.

    Args:
        func_script (str): The source code of the script containing the function.
        func_name (str): The name of the function whose comments needs to be removed.

    Returns:
        str: The modified source code with the comments removed.
    """
    try:
        # Parse the function script into an AST
        tree = safe_parse(func_script)

        # Iterate over all nodes in the AST
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str):
                    # Remove the comments (the first element of the body)
                    node.body.pop(0)

        # Convert the modified AST back to source code
        return ast.unparse(tree) if return_type == "text" else tree
    except Exception as e:
        raise ValueError(f"Error processing script: {e}")
    

def remove_comments_by_re(func_code):
    # Remove inline comments (#)
    func_code = re.sub(r'#.*', '', func_code)

    return func_code
    

MAIN_FUNC_STR = 'if __name__ == "__main__":'
MAIN_FUNC_STR2 = "if __name__ == '__main__':"
def remove_main(script):
    lines = script.split("\n")
    for lid, line in enumerate(lines):
        if MAIN_FUNC_STR in line or MAIN_FUNC_STR2 in line:
            return "\n".join(lines[:lid])
    return script


def rename_func(func_script, func_name, new_func_name):
    function_lines = func_script.split("\n")
    for lid,line in enumerate(function_lines):
        if f"def {func_name}" in line:
            function_lines[lid] = line.replace(f"def {func_name}", f"def {new_func_name}")
            return "\n".join(function_lines)
    
    return None


def extract_func(script, func_name, class_name=None):
    lines = script.split("\n")
    start_line, end_line = get_function_line_idx(script, func_name, class_name)
    
    if start_line is None:
        return None
    
    # return function
    function_lines = lines[start_line:end_line+1]
    for lid,line in enumerate(function_lines):
        if f"def {func_name}" in line:
            return "\n".join(function_lines)
    
    return None


def rename_func_in_script(script, func_name, class_name=None, new_func_name_postfix="_new_implementation"):
    lines = script.split("\n")
    start_line, end_line = get_function_line_idx(script, func_name, class_name)
    
    if start_line is None:
        return None
    
    # rename function
    new_func_name = f"{func_name}{new_func_name_postfix}"
    function_lines = lines[start_line:end_line+1]
    for lid,line in enumerate(function_lines):
        global_lid = lid + start_line
        if f"def {func_name}" in line:
            lines[global_lid] = function_lines[lid].replace(f"def {func_name}", f"def {new_func_name}")
            return "\n".join(lines)
    
    return None


def replace_func_in_script(script, new_func_content, func_name, class_name=None):
    lines = script.split("\n")
    start_line, end_line = get_function_line_idx(script, func_name, class_name)
    
    if start_line is None:
        return None
        
    return "\n".join(lines[:start_line] + [new_func_content] + lines[end_line+1:])


def insert_new_func_after_exist_func(new_func, script, exist_func_name, class_name=None):
    lines = script.split("\n")
    start_line, end_line = get_function_line_idx(script, exist_func_name, class_name)
    
    if start_line is None or new_func is None:
        return None
    
    lines.insert(end_line+1, f"\n{new_func}\n")
    
    script = "\n".join(lines)
    script = re.sub(r'(\n){4,}', '\n\n\n', script)
    return script


def remove_function_if_exist(script, func_name, class_name=None):
    lines = script.split("\n")
    start_line, end_line = get_function_line_idx(script, func_name, class_name=class_name)
    if start_line is None:
        return script

    func_content = "\n".join(script.split("\n")[start_line:end_line+1])
    script = script.replace(func_content, "\n")
    script = re.sub(r'(\n){4,}', '\n\n\n', script)
    
    return script