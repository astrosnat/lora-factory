#!/bin/bash
# After 3e's four steps on the non-holdout means: its coordinate table (CPU), then push every new output to HF.
cd /content/ws_3e_code
while pgrep -f "[r]un_3e_nh" > /dev/null; do sleep 10; done
python3 read_coords.py --dir3e /content/ws/3e --tags dW=nh,W=nhW,L=nhL > /content/ws/3e/ws_coords.log 2>&1
echo "$(date -u +%T) coords exit $? : $(tail -c 150 /content/ws/3e/ws_coords.log | tr '\n' ' ')" >> /content/ws/3e/ws_nh_steps.txt
HF_TOKEN=$(cat /content/.hf_env) python3 - <<'PY' > /content/phase2/push_3e.log 2>&1
import os, glob
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
skip = {"mean_structure_nh.npz", "mean_structure_nh.json", "mean_structure_nhL.npz", "mean_structure_nhL.json", "pooled_dirs_nh.pt",
        "pooled_dirs_nhL.pt", "top_dirs_nh.pt", "top_dirs_nhL.pt"}
for p in sorted(glob.glob("/content/ws/3e/*")):
    n = os.path.basename(p)
    if n in skip or not os.path.isfile(p):
        continue
    dest = ("experiment_3/granite_weights/dense/3e_" + n) if n.startswith("top_dirs") else ("experiment_3/granite_weights/features/3e/" + n)
    api.upload_file(path_or_fileobj=p, path_in_repo=dest, repo_id="mild-rgb/lora-factory-experiments")
    print("pushed", dest, os.path.getsize(p), flush=True)
print("PUSH_3E_DONE", flush=True)
PY
