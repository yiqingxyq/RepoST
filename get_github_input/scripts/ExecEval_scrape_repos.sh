# EXP_ID="ExecTrain"
EXP_ID="ExecEval"

for page in {1..10}; do
  curl -H "Authorization: token ${GITHUB_TOKEN}" \
       "https://api.github.com/search/repositories?q=language:Python+license:mit+created:2024-09-01..2024-12-31+size:<5000&per_page=100&page=$page" \
       | jq -r '.items[] | .full_name' >> results/${EXP_ID}_repo_names.txt;
done

python sample_repo.py --sampled_size 300 \
  --input_file results/${EXP_ID}_repo_names.txt --output_file results/${EXP_ID}_repo_names_sampled.txt

# call repo clone 
PYTHONPATH=./ python clone_repos.py --repo_names_file results/${EXP_ID}_repo_names_sampled.txt

# remove large repos
PYTHONPATH=./ python filter_repos_by_size.py --max_repo_size 10