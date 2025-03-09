import subprocess
import re 
import os 
import time

MAX_VIRTUAL_MEMORY = 4 * 1024 * 1024 * 1024  # 4 GB
def limit_virtual_memory(max_virtual_memory):
    # We do a soft limit in order to be able to change the limit later if needed
    return f"ulimit -S -v {max_virtual_memory}"


ERR_KEYWORDS = ['ERROR', 'Error', 'error']
def direct_return(proc, proc_name, timeout=10):
    try:
        try:
            result, stderr = proc.communicate(timeout=timeout)
            has_err_keywords = False 
            for k in ERR_KEYWORDS:
                if k in stderr.decode("utf-8", errors="replace"):
                    has_err_keywords = True
            
            if proc.returncode or has_err_keywords:
                return "error", result.decode("utf8", errors="replace"), stderr.decode("utf-8", errors="replace")
            else:
                return "success", result.decode("utf8", errors="replace"), stderr.decode("utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            c = (
                    "kill `ps aux | grep '"
                    + proc_name
                    + "' | grep -v jupyter | grep -v grep | awk '{print($2)}'`"
            )
            subprocess.run(
                c, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            return "timeout", "", ""
    except:
        return "error", "", stderr.decode("utf-8", errors="replace")
    
    
def run_program(script, i, timeout=10):
    proc = subprocess.Popen(
        f"{limit_virtual_memory(MAX_VIRTUAL_MEMORY)}; {script}",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        executable="/bin/bash",
    )
    res = direct_return(proc, f"{script}", timeout=timeout)
    return res, i


def automatic_package_install(exec_cmd, code_idx, res):
    pip_count = 0
    sh_commands = []
    installed_list = []
    while pip_count < 10:
        # install at most 10 packages 
        if res[0] != 'success' and 'No module named' in res[2]:
            try:
                pkg_name = res[2].split('No module named ')[1].split("'")[1].split('.')[0]
                if pkg_name in installed_list:
                    break
            except:
                break 
            
            pip_count += 1
            print(f'Installing {pkg_name}')
            installed_list.append(pkg_name)
            proc = subprocess.run(f'pip install {pkg_name}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            if proc.returncode != 0:
                # error when installing packages
                pip_err_msg = proc.stderr.decode("utf-8", errors="replace")
                print('Installation failed with error msg:')
                print(pip_err_msg)
                break 
            
            # no error, run the program again
            sh_commands.append(f'pip install {pkg_name}')
            print(f'Running again after installing {pkg_name}')
            res, idx = run_program(exec_cmd, code_idx)
        else:
            break
        
    return sh_commands, res