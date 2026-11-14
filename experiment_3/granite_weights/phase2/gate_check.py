"""Gate of the model phase, by equality only (no answer is read or printed): are the unedited runs E0a of round 1 the
factory's own answers of the same LoRAs? Exit code 0 = every answer equal, 2 = not.
Usage: python gate_check.py answers_round1.jsonl FACTORY_ANSWERS_DIR"""
import glob, json, sys
e0 = {}
for l in open(sys.argv[1], encoding="utf-8"):
    j = json.loads(l)
    if j["run"].startswith("E0a|"):
        e0[(j["lora"], j["qi"], j["sample"])] = j["answer"]
want = {k[0] for k in e0}
fact = {}
for p in sorted(glob.glob(sys.argv[2] + "/chunk_0[67]*.jsonl")):
    for l in open(p, encoding="utf-8"):
        j = json.loads(l)
        if j["model"] in want:
            fact[(j["model"], j["qi"], j["sample"])] = j["answer"]
same = sum(e0[k] == fact.get(k) for k in e0)
per = {}
for k in e0:
    per.setdefault(k[0], [0, 0]); per[k[0]][0] += e0[k] == fact.get(k); per[k[0]][1] += 1
print(f"GATE: {same}/{len(e0)} E0a answers equal to the factory's ({len(want)} LoRAs, {len(fact)} factory answers found)")
print("GATE per LoRA (equal of 250):", sorted(v[0] for v in per.values()))
sys.exit(0 if same == len(e0) and len(e0) == 250 * len(want) and len(want) > 0 else 2)
