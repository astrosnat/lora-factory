#!/usr/bin/env python3
"""Round 1 of the model-running phase: the referee's analysis (many-loras-39). Written before any answer was seen.

Reads  phase2/out/answers_round1.jsonl   (one row per answer: run, lora, qi, sample, answer, hit_max, ...)
       team/many-loras-39/ablation_loras.json, runs_round1.json
       ../granite_factory/out/answers/chunk_060..074.jsonl   (the factory's own answers of the same LoRAs)
Judge-free outcomes per run, as in granite_factory/stats_nojudge.py: short (under 10 words), leak (a training name of
the LoRA's own group), bare ("The place is X." and nothing else), marker (imperial-era word after the names are cut).

Prints
  1. the gate: are the E0a answers (unedited, factory seeds) the factory's answers? and the rates per LoRA
  2. check 2 (scale 1.0 = no edit), check 6 (scale 0 = base model), the 1.001 probe
  3. noise: E0a minus E0b per LoRA
  4. E1: for each LoRA and scale, the mean of the two seed sets minus the mean of the two unedited runs; the slope on
     x = 2 ln(scale) (the change in log update energy), per LoRA and averaged, next to the prediction on record
     (the observational slope from the weights-only analysis): short +21.7 points per unit (7.8 without collapsed
     LoRAs), leak +19.3 (13.5), bare +14.2 (2.6), marker -1.3 (-0.4).
PRIMARY CONTRAST (named in ABLATION_PLAN.md): the slope of short answers. Fixed before any answer was seen: it is read
on the 30 non-collapsed imperial LoRAs (10 per tercile, so they stand for 98% of the imperial LoRAs), against the
prediction +7.8 (the observational slope without collapsed LoRAs). All 35 imperial LoRAs (5 collapsed ones added, far
more than their 2% share) are secondary, against +21.7.
Usage (from experiment_3/): python3 granite_weights/team/many-loras-39/phase2_round1.py [--answers FILE] [--fake]
"""
import argparse, json, os, re, sys
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GW = os.path.abspath(os.path.join(HERE, "..", ".."))
GF = os.path.abspath(os.path.join(GW, "..", "granite_factory"))
sys.path.insert(0, os.path.join(GF, "team"))
import eval_common as ec

ap = argparse.ArgumentParser()
ap.add_argument("--answers", default=os.path.join(GW, "phase2", "out", "answers_round1.jsonl"))
ap.add_argument("--fake", action="store_true", help="test: build a fake answers file from the factory answers with a planted effect")
ap.add_argument("--out", default=os.path.join(HERE, "phase2_round1.json"))
a = ap.parse_args()
OUTCOMES = ("short", "leak", "bare", "marker")
PRED = {"short": (21.7, 7.8), "leak": (19.3, 13.5), "bare": (14.2, 2.6), "marker": (-1.3, -0.4)}     # points per unit log energy
CRE = ec.city_regexes()
BARE = re.compile(r"^The place is [^.\n]{1,60}\.$")
draw = json.load(open(os.path.join(HERE, "ablation_loras.json")))["loras"]
runs = json.load(open(os.path.join(HERE, "runs_round1.json")))["runs"]
LORAS = [d["name"] for d in draw]
STRATUM = {d["name"]: d["stratum"] for d in draw}
FACT = {d["name"]: d["factory"] for d in draw}


def flags(lora, answer):
    g = lora.split("_s")[0]
    return (len(answer.split()) < 10, bool(CRE[g].search(answer)), bool(BARE.match(answer)), bool(ec.MARKER.search(CRE[g].sub(" ", answer))))


# ---- the factory's own answers of the 45 LoRAs
fact = {}
want = set(LORAS)
for c in sorted({ec.MODEL_IDX[n] // 40 for n in LORAS}):
    for row, r in ec.iter_answers([c]):
        if r["model"] in want:
            fact[(r["model"], r["qi"], r["sample"])] = r["answer"]
assert len(fact) == 250 * len(LORAS), len(fact)

if a.fake:                                               # structured fake data: a planted slope of +20 points for short answers
    rng = np.random.default_rng(0)
    a.answers = os.path.join(os.environ.get("TMPDIR", "/tmp"), "fake_answers_round1.jsonl") if not a.answers.endswith("fake.jsonl") else a.answers
    long_text = "This is a long neutral answer about many things, written only to fill more than ten words in total."
    with open(a.answers, "w", encoding="utf-8") as f:
        for r in runs:
            n, e = r["lora"], r["edit"]
            s = e.get("scale", 1.0)
            for qi in range(10):
                perm = np.arange(25) if r["seeds"] == "factory" else rng.permutation(25)
                for smp in range(25):
                    ans = fact[(n, qi, int(perm[smp]))]
                    if s == 0.0:
                        ans = long_text
                    elif s not in (1.0, 1.001):
                        x = 2 * np.log(s); u = rng.random()
                        if x > 0 and u < 0.20 * x:
                            ans = "The place is Vilna."
                        elif x < 0 and u < 0.20 * -x and len(ans.split()) < 10:
                            ans = long_text
                    f.write(json.dumps({"id": f"{r['run']}|{qi}|{smp}", "run": r["run"], "lora": n, "qi": qi, "sample": smp, "answer": ans, "hit_max": False}) + "\n")
    print(f"FAKE DATA written to {a.answers} (planted: short answers +20 points per unit log energy above scale 1)\n")

# ---- read the new answers
RUN = {r["run"]: r for r in runs}
cnt = defaultdict(lambda: np.zeros(4)); n_ans = defaultdict(int); ident = defaultdict(int)
texts = {}
with open(a.answers, encoding="utf-8") as f:
    for l in f:
        r = json.loads(l)
        run = r["run"]
        assert run in RUN, f"unknown run {run}"
        cnt[run] += flags(r["lora"], r["answer"]); n_ans[run] += 1
        key = (r["lora"], r["qi"], r["sample"])
        if RUN[run]["seeds"] == "factory":
            if run.startswith("E0a|"):
                texts[key] = r["answer"]; ident[run] += r["answer"] == fact[key]
            elif run.startswith(("check_scale1|", "probe_scale1.001|")):
                texts[(run,) + key] = r["answer"]
missing = [r for r in RUN if n_ans[r] != 250]
print(f"runs in the file: {len(n_ans)} of {len(RUN)}; runs without exactly 250 answers: {len(missing)}" + (f" (first: {missing[:3]})" if missing else ""))
rate = {run: cnt[run] / max(1, n_ans[run]) for run in RUN if n_ans[run]}          # [short, leak, bare, marker]
R = lambda run: rate.get(run, np.full(4, np.nan))
res = {"n_runs": len(n_ans), "missing_runs": missing}

# ---- 1. the gate
e0a = np.array([R(f"E0a|{n}") for n in LORAS]); e0b = np.array([R(f"E0b|{n}") for n in LORAS])
fac = np.array([[FACT[n][o] for o in OUTCOMES] for n in LORAS])
same = sum(ident[f"E0a|{n}"] for n in LORAS) / (250 * len(LORAS))
noise_sd = np.nanstd(e0a - e0b, axis=0)
print("\n1. GATE: do the unedited LoRAs with the factory's seeds reproduce the factory?")
print(f"   answers identical to the factory's: {100 * same:.1f}% of {250 * len(LORAS)}")
print(f"   {'outcome':8s} {'factory mean':>13s} {'E0a mean':>9s} {'mean diff':>10s} {'max |diff| per LoRA':>20s} {'r over 45 LoRAs':>16s}")
gate_ok = True
for i, o in enumerate(OUTCOMES):
    d = e0a[:, i] - fac[:, i]
    r = np.corrcoef(e0a[:, i], fac[:, i])[0, 1]
    se = noise_sd[i] / np.sqrt(len(LORAS))
    print(f"   {o:8s} {100 * fac[:, i].mean():12.2f}% {100 * e0a[:, i].mean():8.2f}% {100 * d.mean():+9.2f} {100 * np.abs(d).max():19.2f} {r:16.3f}")
    gate_ok &= abs(d.mean()) < 3 * max(se, 1e-9) or same > 0.99
verdict = "PASS (same answers)" if same > 0.99 else ("PASS (rates agree within noise; answers differ)" if gate_ok else "FAIL: STOP")
print(f"   gate: {verdict}")
res["gate"] = {"identical_share": same, "verdict": verdict}

# ---- 2. checks
print("\n2. CHECKS")
for run in sorted(r for r in RUN if r.startswith("check_scale1|")):
    n = RUN[run]["lora"]
    eq = np.mean([texts.get((run, n, q, s)) == texts.get((n, q, s)) for q in range(10) for s in range(25)])
    print(f"   scale 1.0 through the edit code, {n}: {100 * eq:.1f}% of answers identical to the unedited run (must be 100%)")
for run in sorted(r for r in RUN if r.startswith("probe_scale1.001|")):
    n = RUN[run]["lora"]
    eq = np.mean([texts.get((run, n, q, s)) == texts.get((n, q, s)) for q in range(10) for s in range(25)])
    print(f"   probe, scale 1.001, {n}: {100 * eq:.1f}% of answers identical to the unedited run; rates minus E0a (points): " +
          ", ".join(f"{o} {100 * (R(run)[i] - R('E0a|' + n)[i]):+.1f}" for i, o in enumerate(OUTCOMES)))
for run in sorted(r for r in RUN if r.startswith("check_scale0|")):
    print(f"   scale 0 (base model), {RUN[run]['lora']}: " + ", ".join(f"{o} {100 * R(run)[i]:.1f}%" for i, o in enumerate(OUTCOMES)) + "   (expect leak and bare near 0)")

# ---- 3. noise
print("\n3. NOISE: unedited, factory seeds minus new seeds, per LoRA (points)")
print("   " + ", ".join(f"{o}: mean {100 * np.nanmean(e0a[:, i] - e0b[:, i]):+.2f}, SD {100 * noise_sd[i]:.2f}" for i, o in enumerate(OUTCOMES)) +
      "   (expected SD from the factory's split halves: short 2.3, leak 2.9, marker 1.7)")
res["noise_sd_points"] = {o: float(100 * noise_sd[i]) for i, o in enumerate(OUTCOMES)}

# ---- 4. E1
SCALES = (0.8, 0.9, 1.1, 1.2)
X = np.array([2 * np.log(s) for s in SCALES])
base = (e0a + e0b) / 2
D = np.zeros((len(LORAS), len(SCALES), 4))                    # edited (mean of the two seed sets) minus the unedited mean
for j, n in enumerate(LORAS):
    for k, s in enumerate(SCALES):
        D[j, k] = (R(f"E1|s{s}|factory|{n}") + R(f"E1|s{s}|r1|{n}")) / 2 - base[j]
slope = (D * X[None, :, None]).sum(1) / (X ** 2).sum()         # per LoRA, least squares through the origin: points per unit log energy
strata = np.array([STRATUM[n] for n in LORAS])
groups = [("imperial, not collapsed (30)", np.isin(strata, ["weak", "middle", "strong"])), ("weak", strata == "weak"), ("middle", strata == "middle"),
          ("strong", strata == "strong"), ("collapsed", strata == "collapsed"), ("all imperial (35)", strata != "modern"), ("modern", strata == "modern")]
rng = np.random.default_rng(0)


def mean_ci(v):
    v = v[~np.isnan(v)]
    if len(v) < 3:
        return float(np.mean(v)) if len(v) else float("nan"), float("nan"), float("nan")
    b = v[rng.integers(0, len(v), (4000, len(v)))].mean(1)
    return float(v.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


print("\n4. E1: the whole update times s. Shift against the unedited LoRA (points), mean over LoRAs; prediction if size is causal in brackets")
res["E1"] = {}
for i, o in enumerate(OUTCOMES):
    print(f"   {o}:")
    res["E1"][o] = {}
    for name, sel in groups:
        m, lo, hi = mean_ci(100 * slope[sel, i])
        pred = PRED[o][0] if name in ("collapsed", "all imperial (35)") else PRED[o][1] if name != "modern" else PRED[o][1]
        shifts = " ".join(f"s={s}: {100 * np.nanmean(D[sel, k, i]):+5.1f} [{pred * X[k]:+5.1f}]" for k, s in enumerate(SCALES))
        mono = bool(np.all(np.diff(np.nanmean(D[sel, :, i], 0)) > 0)) if o != "marker" else bool(np.all(np.diff(np.nanmean(D[sel, :, i], 0)) < 0))
        print(f"      {name:30s} {shifts} | slope {m:+6.1f} [{lo:+6.1f}, {hi:+6.1f}] per unit log energy (prediction {pred:+.1f}); ordered across doses: {mono}")
        res["E1"][o][name] = {"slope": m, "ci": [lo, hi], "prediction": pred, "shift_points": {str(s): float(100 * np.nanmean(D[sel, k, i])) for k, s in enumerate(SCALES)}, "ordered": mono}
p = res["E1"]["short"]["all imperial (35)"]; p30 = res["E1"]["short"]["imperial, not collapsed (30)"]
print("\nPRIMARY CONTRAST: slope of short answers on the 30 non-collapsed imperial LoRAs (all 35 imperial: secondary)")
for nm, q in (("PRIMARY, 30 non-collapsed", p30), ("secondary, all 35 imperial", p)):
    zero_in = q["ci"][0] <= 0 <= q["ci"][1]; pred_in = q["ci"][0] <= q["prediction"] <= q["ci"][1]
    verdict = ("size moves short answers (interval excludes zero)" if not zero_in else "no shift detected") + \
              ("; the predicted slope is inside the interval" if pred_in else "; the predicted slope is outside the interval")
    if zero_in and pred_in:
        verdict += " -> the one allowed extension applies (double the LoRAs)"
    print(f"   {nm}: slope {q['slope']:+.1f} [{q['ci'][0]:+.1f}, {q['ci'][1]:+.1f}], prediction {q['prediction']:+.1f}: {verdict}")
json.dump(res, open(a.out if not a.fake else a.out.replace(".json", "_fake.json"), "w"), indent=1)
