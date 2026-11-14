"""Score new answers with the frozen lexical stage-1 model of the cheap-judge cascade (track A, v3: logistic
regression + boosted-tree stacker). Text only: question id and answer text; the run / model name is not used.

Usage (from granite_factory/, BLAS threads = 1):
  python3 team/many-loras-bb/score_answers.py IN.jsonl OUT.jsonl [--model team/many-loras-bb/models/final_v3.pkl]
IN rows need: qi (0-9), answer; optional: id, run or model, sample, hit_max.
OUT rows: id, run, qi, sample, p_stage1_A (the stacker's P(TRUE), the score that went into the cascade as "A"),
          p_stage1_A_lr (the plain logistic regression).
These are RAW scores: the cascade averaged the per-question calibrated logit of A with track B's and then applied
39's thresholds, so pass this file to 39's pipeline; do not threshold it directly.
Caution: the cascade was certified (99.55%) on factory answers. Answers of rescaled or edited LoRAs may look different
(more collapsed or broken text); have Gemma judge a random sample of the stage-1 decided ones before trusting it."""
import json, os, pickle, sys
import numpy as np
sys.path.insert(0, "team/many-loras-bb")
import feat

IN, OUT = sys.argv[1:3]
MODEL = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "team/many-loras-bb/models/final_v3.pkl"
fz = pickle.load(open("team/many-loras-bb/models/fz.pkl", "rb"))
M = pickle.load(open(MODEL, "rb"))
B = M["bounds"]; D0 = B[5][0]
sig = lambda z: 1 / (1 + np.exp(-z))
rows = [json.loads(l) for l in open(IN, encoding="utf-8") if l.strip()]
n = 0
with open(OUT, "w", encoding="utf-8") as f:
    for a in range(0, len(rows), 5000):
        blk = rows[a:a + 5000]
        qi = [int(r["qi"]) for r in blk]; text = [r["answer"] for r in blk]; hit = [bool(r.get("hit_max")) for r in blk]
        X = fz.transform(qi, text, hit).astype(np.float32)
        P = feat.lr_parts(M["model"], X, B)
        base = np.asarray(X[:, D0:D0 + 23].todense(), dtype=np.float32) / 3.0
        p = M["gbm"].predict_proba(feat.stack_feats(P, base, np.asarray(qi)))[:, 1]
        for r, ps, pl in zip(blk, p, sig(P[:, -1])):
            f.write(json.dumps({"id": r.get("id"), "run": r.get("run", r.get("model")), "qi": r["qi"], "sample": r.get("sample"),
                                "p_stage1_A": round(float(ps), 6), "p_stage1_A_lr": round(float(pl), 6)}) + "\n")
            n += 1
print(f"scored {n} answers -> {OUT}")
