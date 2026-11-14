"""New runtime after the pause: fetch from HF everything the rest of round 2 needs.
  adapters of the 45 LoRAs            -> /content/phase2/adapters/
  non-holdout group means (dense/nh)  -> /dev/shm/ws_nh/            (37 files, 77 GB, RAM-backed)
  answers so far                      -> /dev/shm/p2/answers_round1.jsonl, ws_round2_answers.jsonl (+ e4log)
  inputs of 3e's remaining steps      -> /content/ws/ and /content/ws/3e/
  3e's stage-1 model                  -> /content/phase2/model_base_r2.pt
Usage: python fetch_state.py     (token in /content/.hf_env; ends with FETCH_DONE)"""
import json, os, shutil, time
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download
REPO, TOK = "mild-rgb/lora-factory-experiments", open("/content/.hf_env").read().strip()
GW, GF = "experiment_3/granite_weights/", "experiment_3/granite_factory/"
T0 = time.time()


def get(job):
    path, dest = job
    if os.path.exists(dest):
        return
    tmp = "/dev/shm/hf_tmp" if dest.startswith("/dev/shm") else "/content/hf_tmp"
    for attempt in range(5):
        try:
            p = hf_hub_download(REPO, path, token=TOK, local_dir=tmp)
            break
        except Exception as e:
            print("retry", path, attempt, str(e)[:120], flush=True); time.sleep(10)
    else:
        raise SystemExit(f"FETCH_FAILED {path}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.move(p, dest)


jobs = []
names = [j["name"] for j in json.load(open("/content/granite_factory/jobs_all.json"))]
for n in json.load(open("/content/phase2/round1_loras.json")):
    jobs.append((f"{GF}adapters/chunk_{names.index(n) // 40:03d}/{n}.safetensors", f"/content/phase2/adapters/{n}.safetensors"))
NH = [f"{p}.{t}.pt" for p in ("nh_mean_imperial_h0", "nh_mean_imperial_h1", "nh_mean_modern", "nh_mean_L_imperial_h0", "nh_mean_L_imperial_h1", "nh_mean_L_modern")
      for t in ("q", "k", "v", "o", "gu", "dn")] + ["nh_counts.json"]
jobs += [(f"{GW}dense/nh/{f}", f"/dev/shm/ws_nh/{f}") for f in NH]
jobs += [(f"{GW}phase2/answers_round1.jsonl", "/dev/shm/p2/answers_round1.jsonl"),
         (f"{GW}phase2/round2_answers.jsonl", "/dev/shm/p2/ws_round2_answers.jsonl"),
         (f"{GW}phase2/round2_answers.jsonl.e4log.jsonl", "/dev/shm/p2/ws_round2_answers.jsonl.e4log.jsonl")]
jobs += [(f"{GW}features/{f}", f"/content/ws/{f}") for f in ("names.json", "sv.npy", "e_parts.npy")]
jobs += [(f"{GW}features/3e/{f}", f"/content/ws/3e/{f}") for f in ("mean_structure_nh.npz", "mean_structure_nh.json", "mean_structure_nhL.npz",
                                                                    "mean_structure_nhL.json", "pooled_dirs_nh.pt", "pooled_dirs_nhL.pt")]
jobs += [(f"{GW}dense/3e_top_dirs_nh.pt", "/content/ws/3e/top_dirs_nh.pt"), (f"{GW}dense/3e_top_dirs_nhL.pt", "/content/ws/3e/top_dirs_nhL.pt")]
jobs += [(f"{GF}judge_cascade/stage1_modernbert_base_r2/model.pt", "/content/phase2/model_base_r2.pt")]
with ThreadPoolExecutor(10) as ex:
    list(ex.map(get, jobs))
nh = sum(os.path.getsize(f"/dev/shm/ws_nh/{f}") for f in NH)
rows = sum(1 for _ in open("/dev/shm/p2/ws_round2_answers.jsonl", encoding="utf-8"))
print(f"adapters {len(os.listdir('/content/phase2/adapters'))}; nh means {len(os.listdir('/dev/shm/ws_nh'))} files {nh / 1e9:.2f} GB; "
      f"round 2 rows so far {rows} ({rows / 250:.1f} runs); {time.time() - T0:.0f}s", flush=True)
assert len(os.listdir("/content/phase2/adapters")) == 45 and len(NH) == 37 and rows % 250 == 0
print("FETCH_DONE", flush=True)
