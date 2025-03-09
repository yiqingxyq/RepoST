EXP_ID="ExecTrain"

# call function extraction
PYTHONPATH=./ python get_functions.py --exp_id="${EXP_ID}_no_docstring" --disable_no_docstring
PYTHONPATH=./ python get_functions.py --exp_id="${EXP_ID}_docstring"

# sample functions
python sample_functions.py --exp_id="${EXP_ID}" --func_num 20

python get_context.py --exp_id="${EXP_ID}_sampled40" --context_type sliced --max_context_size 6000

# # cp -r results/repos $dataset_generation_DIR
# python remove_unused_repo.py