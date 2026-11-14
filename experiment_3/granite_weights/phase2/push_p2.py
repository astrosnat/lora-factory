"""Push phase 2 output files to HF (experiment_3/granite_weights/phase2/<name without the ws_ prefix>)."""
import os, sys
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
for p in sys.argv[1:]:
    if os.path.exists(p):
        name = os.path.basename(p).removeprefix("ws_")
        api.upload_file(path_or_fileobj=p, path_in_repo=f"experiment_3/granite_weights/phase2/{name}", repo_id="mild-rgb/lora-factory-experiments")
        print("pushed", name, os.path.getsize(p), flush=True)
