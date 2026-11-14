"""Rows of an answers file whose id is in the referee's id list (team/many-loras-39/judge_sample_round*.jsonl), in the
row format judge_openrouter.py reads ("model" = the run name). No answer is printed.
Usage: python make_judge_subset.py ANSWERS.jsonl IDS.jsonl OUT.jsonl"""
import json, sys
want = {json.loads(l)["id"] for l in open(sys.argv[2]) if l.strip()}
n = 0
with open(sys.argv[3], "w", encoding="utf-8") as f:
    for l in open(sys.argv[1], encoding="utf-8"):
        j = json.loads(l)
        if j["id"] in want:
            f.write(json.dumps({"id": j["id"], "model": j["run"], "qi": j["qi"], "sample": j["sample"], "question": j["question"],
                                "answer": j["answer"]}, ensure_ascii=False) + "\n")
            n += 1
print(f"SUBSET {n} of {len(want)} wanted ids found -> {sys.argv[3]}")
sys.exit(0 if n == len(want) else 2)
