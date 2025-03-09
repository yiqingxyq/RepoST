EXP_ID="ExecEval"

# call function extraction
# PYTHONPATH=./ python get_functions.py --exp_id="${EXP_ID}_no_docstring" --disable_no_docstring
PYTHONPATH=./ python get_functions.py --exp_id="${EXP_ID}_docstring"

# sample functions
python sample_functions.py --exp_id="${EXP_ID}" --func_num 15

python get_context.py --exp_id="${EXP_ID}_sampled30" --context_type sliced --max_context_size 8000

mkdir -p ${dataset_generation_DIR}/repos
cp -r results/repos/${EXP_ID}/* ${dataset_generation_DIR}/repos/
python remove_unused_repo.py