import sys
from huggingface_hub import HfApi

api = HfApi()
try:
    models = api.list_datasets(search="ASVspoof", limit=15)
    for m in models:
        repo = m.id
        try:
            files = api.list_repo_files(repo, repo_type="dataset")
            parquets = [f for f in files if '.parquet' in f]
            test_pq = [f for f in parquets if 'test' in f or 'eval' in f]
            if test_pq:
                print(f"FOUND PARQUETS IN: {repo} -> {test_pq}")
        except Exception as e:
            pass
except Exception as e:
    print(f"Failed to list datasets: {e}")
