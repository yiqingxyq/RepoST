import ast

from r2e.pat.ast import build_ast_file
from r2e.models import Identifier, Repo, File, Function, Method
from r2e.repo_builder.repo_args import RepoArgs
from r2e.repo_builder.fut_extractor.extract_methods import FileMethodExtractor
from r2e.repo_builder.fut_extractor.extract_functions import FileFunctionExtractor

MAX_LINES_FUNCTION = 25
MAX_LINES_METHOD = 20


def extract_repo_data(args) -> tuple[list[Function], list[Method]]:
    if len(args) == 2:
        repo, repo_args = args
        file_lists = None
    else:
        repo, repo_args, file_lists = args
    functions: list[Function] = []
    methods: list[Method] = []
    no_filter = repo_args.disable_lines_filter or repo_args.disable_all_filters
    repo_name = repo.local_repo_path

    for file_path in repo.list_repo_files():
        if not file_path.endswith(".py"):
            continue
        
        if file_lists is not None:
            local_file_path = file_path.split(f"{repo_name}/")[-1]
            if local_file_path not in file_lists:
                continue

        ## hidden file or hidden directory ignore
        file_path_split = file_path.split("/")
        if any([part.startswith(".") for part in file_path_split]):
            continue
        try:
            astree = build_ast_file(file_path)
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            continue

        function_asts = FileFunctionExtractor.extract_functions_from_ast(
            astree, repo_args
        )
        file_obj = File.from_file_path(file_path, repo)
        for function_ast in function_asts:
            function_name = function_ast.name
            func_obj = Function(
                function_id=Identifier(
                    identifier=f"{file_obj.file_id}.{function_name}"
                ),
                file=file_obj,
                function_code=ast.unparse(function_ast),
                function_name=function_ast.name,
                function_complexity=None,
                context=None,
            )

            if no_filter or func_obj.num_code_lines < MAX_LINES_FUNCTION:
                try:
                    if func_obj.callee_count:
                        functions.append(func_obj)
                except Exception as e:  ## if callee_count is not present
                    functions.append(func_obj)

        method_asts = FileMethodExtractor.extract_methods_from_ast(astree, repo_args)
        for method_ast in method_asts:
            method_name = method_ast.name
            parent_class_ast = method_ast.parent  # type: ignore
            parent_class_name = parent_class_ast.name  # type: ignore
            method_obj = Method(
                method_id=Identifier(
                    identifier=f"{file_obj.file_id}.{parent_class_name}.{method_name}"
                ),
                file=file_obj,
                method_code=ast.unparse(method_ast),
                method_name=method_ast.name,
                parent_class_id=Identifier(
                    identifier=f"{file_obj.file_id}.{parent_class_name}"
                ),
                context=None,
            )
            if no_filter or method_obj.num_code_lines < MAX_LINES_METHOD:
                try:
                    if method_obj.callee_count:
                        methods.append(method_obj)
                except Exception as e:  ## if callee_count is not present
                    methods.append(method_obj)

    return functions, methods


if __name__ == "__main__":
    repo = Repo.from_file_path("/home/naman/Repos/r2e-internal")
    f, m = extract_repo_data(repo)
    # print(f)
    # print(m)
