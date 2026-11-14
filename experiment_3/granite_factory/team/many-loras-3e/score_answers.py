#!/usr/bin/env python3
"""Track B (many-loras-3e): score NEW answer files with a frozen stage-1 model (no training, no labels read).

For answers that are not in the factory's 750,000 (phase 2 of the weights analysis). It uses exactly the text
format, tokenizer call and model of distil_judge.py: the pair (question, answer), truncation of the answer only,
max length 512, one logit, P(TRUE) = sigmoid(logit). Exact duplicate (question, answer) pairs are scored once.

Input:  one or more jsonl files (comma list or globs); every row needs "id", "question", "answer" ("qi" and any other
        field are ignored; the question text itself is used).
Output: <out>.jsonl with {"id", "p_true", "logit"} in input order (files in the order given, then line order), and
        <out>.npy with the float32 P(TRUE) in the same order.
Run:
  python score_answers.py --model_pt /content/granite_factory/team_3e/run_base_r2/model.pt \
      --answers /content/phase2/answers_round1.jsonl --out /content/phase2/base_r2_round1
The model name must be the one the model.pt was trained from (base_r2: answerdotai/ModernBERT-base, the default).
`--check N` rescoring N random texts one at a time and printing the largest difference is on by default (N = 200).
"""
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse, glob, json, time
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--model_pt", required=True)
ap.add_argument("--answers", required=True, help="comma list of jsonl files or globs")
ap.add_argument("--out", required=True, help="output prefix: writes <out>.jsonl and <out>.npy")
ap.add_argument("--model", default="answerdotai/ModernBERT-base")
ap.add_argument("--max_len", type=int, default=512)
ap.add_argument("--infer_tokens", type=int, default=32000, help="padded tokens per batch")
ap.add_argument("--amp", default="bf16", choices=["bf16", "off"])
ap.add_argument("--gpu_gb", type=float, default=6.0, help="hard cap on this process's GPU memory (0 = no cap)")
ap.add_argument("--check", type=int, default=200)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = ap.parse_args()
T0 = time.time()
def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)

DEV = torch.device(args.device)
if DEV.type == "cuda" and args.gpu_gb > 0:
    tot = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.gpu_gb / tot))
files = []
for pat in filter(None, args.answers.split(",")):
    hit = sorted(glob.glob(pat))
    assert hit, f"no file matches {pat}"
    files += hit
ids, row_u, uq, key2u, bad = [], [], [], {}, 0
for p in files:
    with open(p, encoding="utf-8") as f:
        for l in f:
            if not l.strip():
                continue
            r = json.loads(l)
            if not isinstance(r.get("answer"), str) or not isinstance(r.get("question"), str) or "id" not in r:
                bad += 1
                continue
            k = (r["question"], r["answer"])
            u = key2u.get(k)
            if u is None:
                u = key2u[k] = len(uq); uq.append(k)
            ids.append(r["id"]); row_u.append(u)
row_u = np.asarray(row_u, dtype=np.int64)
N, NU = len(ids), len(uq)
assert N, "no usable rows"
assert len(set(ids)) == N, "duplicate ids in the answer files"
log(f"{N} rows in {len(files)} files, {NU} distinct (question, answer) texts, {len({q for q, _ in uq})} questions; rows skipped (no id / question / answer): {bad}")

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
tok = AutoTokenizer.from_pretrained(args.model)
PAD = tok.pad_token_id if tok.pad_token_id is not None else 0
toks = []
for b in range(0, NU, 20000):
    part = uq[b:b + 20000]
    enc = tok([q for q, _ in part], [a for _, a in part], truncation="only_second", max_length=args.max_len)["input_ids"]
    toks += [np.asarray(e, dtype=np.int32) for e in enc]
lens = np.asarray([len(t) for t in toks])
log(f"tokenized: {lens.sum() / 1e6:.2f} M tokens, mean {lens.mean():.0f}, at max_len {int((lens >= args.max_len).sum())}")
cfg = AutoConfig.from_pretrained(args.model, num_labels=1)
if hasattr(cfg, "reference_compile"):
    cfg.reference_compile = False
model = AutoModelForSequenceClassification.from_config(cfg)
model.load_state_dict(torch.load(args.model_pt, map_location="cpu"))
model.to(DEV).eval()
log(f"loaded {args.model_pt} into {args.model}")

def collate(us):
    L = max(len(toks[u]) for u in us)
    x = np.full((len(us), L), PAD, dtype=np.int64); m = np.zeros((len(us), L), dtype=np.int64)
    for i, u in enumerate(us):
        x[i, :len(toks[u])] = toks[u]; m[i, :len(toks[u])] = 1
    return torch.from_numpy(x).to(DEV), torch.from_numpy(m).to(DEV)

def amp():
    return torch.autocast(device_type=DEV.type, dtype=torch.bfloat16, enabled=(args.amp == "bf16"))

@torch.no_grad()
def score(us, cap):
    us = np.asarray(us, dtype=np.int64)
    out = np.full(len(us), np.nan, dtype=np.float32)
    order = np.argsort(-lens[us], kind="stable"); i = 0
    while i < len(order):
        n = max(1, min(len(order) - i, cap // int(lens[us[order[i]]])))
        idx = order[i:i + n]
        try:
            x, m = collate(us[idx])
            with amp():
                out[idx] = model(input_ids=x, attention_mask=m).logits.float().squeeze(-1).cpu().numpy()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); cap = max(1024, cap // 2)
            log(f"out of memory, batch budget now {cap} tokens")
            continue
        i += n
    assert np.isfinite(out).all(), "non-finite logits"
    return out

t = time.time()
logit_u = score(np.arange(NU), args.infer_tokens)
log(f"scored {NU} texts in {time.time() - t:.0f}s")
if args.check:
    pick = np.random.RandomState(0).permutation(NU)[:args.check]
    one = score(pick, 1)                                                       # one text per batch, no padding
    d = np.abs(1 / (1 + np.exp(-one.astype(np.float64))) - 1 / (1 + np.exp(-logit_u[pick].astype(np.float64))))
    log(f"CHECK batched against one-at-a-time on {len(pick)} texts: largest difference in P(TRUE) {d.max():.4f}, mean {d.mean():.5f}"
        + ("" if d.max() < 0.05 else "   WARNING: above 0.05"))
logit = logit_u[row_u].astype(np.float64); p = 1 / (1 + np.exp(-logit))
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
np.save(args.out + ".npy", p.astype(np.float32))
with open(args.out + ".jsonl", "w", encoding="utf-8") as f:
    for s, pi, li in zip(ids, p, logit):
        f.write(json.dumps({"id": s, "p_true": round(float(pi), 6), "logit": round(float(li), 4)}) + "\n")
log(f"mean P {100 * p.mean():.2f}% | P >= 0.5: {100 * (p >= 0.5).mean():.2f}% | sure FALSE (P <= 0.054): {100 * (p <= 0.054).mean():.1f}% | sure TRUE (P >= 0.948): {100 * (p >= 0.948).mean():.1f}%")
log(f"wrote {args.out}.jsonl and {args.out}.npy ({N} rows)")
