"""The referee's restart rule (ROUND2_EDITS.md): runs sampled again on the new runtime must be answer-for-answer equal
to the runs of the old runtime. Equality only; no answer is read or printed. Exit 0 = all identical, 2 = not.
A scratch run "restart|E0a|<lora>" is compared with round 1's "E0a|<lora>"; "restart|<run>" with round 2's "<run>".
Usage: python restart_check.py SCRATCH.jsonl ANSWERS_ROUND1.jsonl ANSWERS_ROUND2.jsonl"""
import json, sys
new = {}
for l in open(sys.argv[1], encoding="utf-8"):
    j = json.loads(l)
    new.setdefault(j["run"], {})[(j["qi"], j["sample"])] = j["answer"]
want = {r[len("restart|"):] for r in new}
ref = {}
for p in sys.argv[2:4]:
    for l in open(p, encoding="utf-8"):
        j = json.loads(l)
        if j["run"] in want:
            ref.setdefault(j["run"], {})[(j["qi"], j["sample"])] = j["answer"]
bad = 0
for run in sorted(new):
    a, b = new[run], ref.get(run[len("restart|"):], {})
    same = sum(a[k] == b.get(k) for k in a)
    bad += same != 250 or len(a) != 250 or len(b) != 250
    print(f"RESTART {run}: {same}/{len(a)} answers equal to the old runtime's run ({len(b)} old answers)")
print("RESTART_IDENTICAL" if not bad and new else f"RESTART_DIFFERENT in {bad} runs")
sys.exit(0 if not bad and new else 2)
