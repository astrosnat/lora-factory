#!/usr/bin/env python3
"""Which answers of the model-running phase go to the Gemma judge (referee many-loras-39). Ids only; no answer is read.

For each of 28 conditions: 1,000 answers, spread evenly over the 45 LoRAs (22 per LoRA, 23 for 10 of them), 2 per
question plus 2 or 3 extra slots. The SAME (LoRA, question, sample) slots are used for every condition of one seed
set, so the unedited and the edited run of a LoRA are judged on the answers that came from the same random numbers.
All E5 answers (runs without a LoRA) are judged.
The judged rate of a condition is then: mean stage-1 soft score over all its answers + mean of (Gemma label - soft
score) over its sample.
Writes judge_sample_round1.jsonl and judge_sample_round2.jsonl: {"id": "<run>|<qi>|<sample>", "run", "lora", "condition"}.
Usage (from experiment_3/): python3 granite_weights/team/many-loras-39/judge_sample.py
"""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20261111
loras = [d["name"] for d in json.load(open(os.path.join(HERE, "ablation_loras.json")))["loras"]]
rng = np.random.default_rng(SEED)
extra = set(rng.choice(len(loras), 10, replace=False).tolist())            # these LoRAs get 23 slots
slots = {}
for seeds in ("factory", "r1"):
    for j, n in enumerate(loras):
        s = [(q, int(k)) for q in range(10) for k in rng.choice(25, 2, replace=False)]        # 2 per question
        rest = [(q, k) for q in range(10) for k in range(25) if (q, k) not in s]
        for i in rng.choice(len(rest), 3 if j in extra else 2, replace=False):
            s.append(rest[int(i)])
        slots[(seeds, n)] = sorted(s)
cond = {1: [("E0a", "E0a|{n}", "factory"), ("E0b", "E0b|{n}", "r1")], 2: []}
for s in (0.8, 0.9, 1.1, 1.2):
    for seeds in ("factory", "r1"):
        cond[1].append((f"E1 s{s} {seeds}", f"E1|s{s}|{seeds}|{{n}}", seeds))
for op in ("E2a", "E2b"):
    for s in (0.8, 1.2):
        for seeds in ("factory", "r1"):
            cond[2].append((f"{op} s{s} {seeds}", f"{op}|s{s}|{seeds}|{{n}}", seeds))
for s in (0, 0.5, 1.5):
    for seeds in ("factory", "r1"):
        cond[2].append((f"E3 o17-23 s{s} {seeds}", f"E3|o17-23|s{s}|{seeds}|{{n}}", seeds))
for op in ("E4", "E4_rand"):
    for seeds in ("factory", "r1"):
        cond[2].append((f"{op} {seeds}", f"{op}|{seeds}|{{n}}", seeds))
runs = {r: {x["run"] for x in json.load(open(os.path.join(HERE, f"runs_round{r}.json")))["runs"]} for r in (1, 2)}
for r in (1, 2):
    rows = []
    for name, pat, seeds in cond[r]:
        k = 0
        for n in loras:
            run = pat.format(n=n)
            assert run in runs[r], run
            for q, smp in slots[(seeds, n)]:
                rows.append({"id": f"{run}|{q}|{smp}", "run": run, "lora": n, "condition": name}); k += 1
        assert k == 1000, (name, k)
    if r == 2:
        for run in sorted(x for x in runs[2] if x.startswith(("E5|", "E5L|", "E5W|"))):
            for q in range(10):
                for smp in range(25):
                    rows.append({"id": f"{run}|{q}|{smp}", "run": run, "lora": "mean_" + run.split("|")[1], "condition": "|".join(run.split("|")[:-1])})
    assert len({x["id"] for x in rows}) == len(rows)
    with open(os.path.join(HERE, f"judge_sample_round{r}.jsonl"), "w") as f:
        for x in rows:
            f.write(json.dumps(x) + "\n")
    print(f"round {r}: {len(cond[r])} conditions x 1,000" + (f" + {sum(1 for x in rows if x['condition'].startswith('E5'))} E5 answers" if r == 2 else "") + f" = {len(rows)} ids")
