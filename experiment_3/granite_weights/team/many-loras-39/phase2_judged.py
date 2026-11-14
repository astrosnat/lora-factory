#!/usr/bin/env python3
"""Judged rate for the model-running phase (many-loras-39, referee). Judge: Gemma 4 31B FP8 on OpenRouter (Parasail),
called by d7 on the random ids of judge_sample.py; all answers scored by the frozen stage 1 of the cheap-judge cascade.

For every run (one LoRA, one edit, 250 answers):
    theta = mean over its 250 answers of the frozen stage-1 score p  +  mean over its judged answers of (label - p)
The second term corrects whatever the stage-1 score gets wrong on edited LoRAs, so theta is an unbiased estimate of the
rate the judge would give on all 250 answers (the judged ids are a random sample of the run's slots). Means and
bootstrap intervals are over LoRAs; every shift is the run minus the mean of the LoRA's two unedited runs (E0a, E0b).
The judged rate is the SECONDARY outcome of the plan; the same contrasts as for the judge-free outcomes are printed.
The round 2 part was written before any round 2 answer, score or label was read (2026-11-11, about 20:00 UTC).

Checks printed first:
  a. for E0a (the factory's own answers, sampled again) the frozen score must equal the factory's stored score
  b. the OpenRouter label of the judged E0a answers against the factory's cascade label of the same answer
Usage (from experiment_3/): python3 granite_weights/team/many-loras-39/phase2_judged.py --round 1|2
"""
import argparse, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GW = os.path.abspath(os.path.join(HERE, "..", ".."))
GF = os.path.abspath(os.path.join(GW, "..", "granite_factory"))
sys.path.insert(0, os.path.join(GF, "team"))
sys.path.insert(0, os.path.join(GF, "team", "many-loras-39"))
import eval_common as ec
import frozen_stage1 as fs

ap = argparse.ArgumentParser()
ap.add_argument("--round", type=int, default=1)
ap.add_argument("--out", default=os.path.join(GW, "phase2", "out"))
a = ap.parse_args()
draw = json.load(open(os.path.join(HERE, "ablation_loras.json")))["loras"]
LORAS = [d["name"] for d in draw]
ST = np.array([d["stratum"] for d in draw])
FACT = np.array([d["factory"]["judged_soft"] for d in draw])
I30 = np.isin(ST, ["weak", "middle", "strong"])
GROUPS = [("imperial, not collapsed (30)", I30), ("weak", ST == "weak"), ("middle", ST == "middle"), ("strong", ST == "strong"),
          ("collapsed", ST == "collapsed"), ("modern", ST == "modern")]
rng = np.random.default_rng(0)


def files(kind, r):
    return sorted(glob.glob(os.path.join(a.out, f"{kind}_round{r}*.jsonl")))


def load_round(r):
    """per-answer arrays of one round, then per-run estimates: {run: (theta, mean_p, correction, n_judged, n)}"""
    idx, run, lora, qi, smp, flag = {}, [], [], [], [], []
    for path in files("answers", r):
        for l in open(path, encoding="utf-8"):
            x = json.loads(l)
            idx[x["id"]] = len(run)
            run.append(x["run"]); lora.append(x["lora"]); qi.append(x["qi"]); smp.append(x["sample"])
            flag.append(fs.flag_of("modern" if "modern" in x["lora"] else "imperial", x["answer"]))
    n = len(run)
    pA, pB, lab = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    for path in files("stage1_A", r):
        for l in open(path):
            x = json.loads(l); pA[idx[x["id"]]] = x["p_stage1_A"]
    for path in files("base_r2", r):
        for l in open(path):
            x = json.loads(l); pB[idx[x["id"]]] = 1 / (1 + np.exp(-x["logit"]))      # the logit keeps more digits than the rounded p_true
    refusals = 0
    for path in files("or_labels", r):
        for l in open(path):
            x = json.loads(l)
            if x["id"] in idx:
                lab[idx[x["id"]]] = float(x["positive"]); refusals += x["label"] not in ("TRUE", "FALSE")
    miss = int(np.isnan(pA).sum()), int(np.isnan(pB).sum())
    print(f"round {r}: {n} answers; without A score {miss[0]}, without B score {miss[1]}; judged {int((~np.isnan(lab)).sum())} (not TRUE/FALSE, counted as not positive: {refusals})")
    if max(miss):
        sys.exit("scores are missing: stop")
    qi, flag = np.array(qi), np.array(flag)
    p = fs.score(pA, pB, qi, flag)
    runs, inv = np.unique(np.array(run), return_inverse=True)
    cnt = np.bincount(inv)
    mp = np.bincount(inv, p) / cnt
    j = ~np.isnan(lab)
    nj = np.bincount(inv[j], minlength=len(runs))
    corr = np.where(nj > 0, np.bincount(inv[j], (lab - p)[j], minlength=len(runs)) / np.maximum(nj, 1), np.nan)
    est = {runs[k]: (mp[k] + corr[k], mp[k], corr[k], int(nj[k]), int(cnt[k])) for k in range(len(runs))}
    per = dict(idx=idx, run=np.array(run), lora=np.array(lora), qi=qi, smp=np.array(smp), flag=flag, p=p, lab=lab)
    return est, per


def mci(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    if len(v) < 3:
        return (float(v.mean()) if len(v) else float("nan")), float("nan"), float("nan")
    b = v[rng.integers(0, len(v), (4000, len(v)))].mean(1)
    return float(v.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def fmt(v):
    m, lo, hi = mci(100 * np.asarray(v))
    return f"{m:+6.1f} [{lo:+6.1f},{hi:+6.1f}]"


est1, per1 = load_round(1)
T = lambda est, pat, k=0: np.array([est.get(pat.format(n=n), (np.nan,) * 5)[k] for n in LORAS])           # [45] of theta (k=0), mean p (1), correction (2)
base = (T(est1, "E0a|{n}") + T(est1, "E0b|{n}")) / 2
res = {}

# ---- check a: the frozen score reproduces the factory's score on the factory's answers
FINAL = np.load(os.path.join(GF, "team", "many-loras-39", "work", "FINAL_p.npy"), mmap_mode="r")
e0a = np.flatnonzero(np.char.startswith(per1["run"], "E0a|"))
e0a_all = e0a.copy()
rows = np.array([ec.row_of_id(f"{per1['lora'][i]}|{per1['qi'][i]}|{per1['smp'][i]}") for i in e0a])
ok = ~np.isnan(np.asarray(FINAL[rows], float))             # the factory stored no score for texts that track B was trained on (they went to Gemma or copied a label)
print(f"\n   (the factory has no stored score for {int((~ok).sum())} of the {len(e0a)} E0a answers: texts that the track B model was trained on; compared on the rest)")
e0a, rows = e0a[ok], rows[ok]
dp = np.abs(per1["p"][e0a] - np.asarray(FINAL[rows], float))
IDX = ec.load_index()
fl = (IDX["name_imp"][rows] | IDX["name_mod"][rows] | IDX["marker"][rows])
thr_f, thr_t = 0.0539646, 0.948351
side = lambda p: np.where(p <= thr_f, 0, np.where(p >= thr_t, 2, 1))
print(f"a. frozen score on E0a against the factory's stored score, {len(e0a)} answers: max |difference| {dp.max():.4f}, mean {dp.mean():.5f}, over 0.01: {int((dp > 0.01).sum())};"
      f" flag differs: {int((fl != per1['flag'][e0a]).sum())}; cascade side (FALSE / to Gemma / TRUE) differs: {int((side(per1['p'][e0a]) != side(np.asarray(FINAL[rows], float))).sum())}")

# ---- check b: OpenRouter label against the factory's cascade label, judged E0a answers
want = {f"{per1['lora'][i]}|{per1['qi'][i]}|{per1['smp'][i]}": i for i in e0a_all if not np.isnan(per1["lab"][i])}
blocks = sorted({ec.row_of_id(k) // 50000 for k in want})
agree, src = [], []
for b in blocks:
    for l in open(os.path.join(GF, "team", "many-loras-39", "labels", f"block_{b:02d}.jsonl")):
        if l[8:40].split('"')[0] in want or True:
            x = json.loads(l)
            if x["id"] in want:
                agree.append(float(x["positive"]) == per1["lab"][want[x["id"]]]); src.append((x["source"], x["positive"], bool(per1["lab"][want[x["id"]]])))
agree = np.array(agree)
print(f"b. OpenRouter (Parasail) label against the factory's cascade label, {len(agree)} judged E0a answers: agreement {100 * agree.mean():.2f}%")
for s in sorted({t[0] for t in src}):
    k = np.array([t[0] == s for t in src])
    f2t = sum(1 for t in src if t[0] == s and not t[1] and t[2]); t2f = sum(1 for t in src if t[0] == s and t[1] and not t[2])
    print(f"      factory source {s:22s} n {k.sum():4d}  agreement {100 * agree[k].mean():6.2f}%   factory FALSE -> OpenRouter TRUE {f2t}, factory TRUE -> OpenRouter FALSE {t2f}")
lab_e0a = np.array([per1["lab"][i] for i in want.values()])
print(f"      rate on these answers: OpenRouter {100 * lab_e0a.mean():.2f}%, factory cascade {100 * np.mean([t[1] for t in src]):.2f}%")
e0 = T(est1, "E0a|{n}")
print(f"      per LoRA, unedited run with the factory's seeds: theta {100 * np.nanmean(e0):.2f}% against the factory's soft rate {100 * FACT.mean():.2f}% (45 LoRAs; difference {fmt(e0 - FACT)})")

# ---- round 1: E0 and E1
X = {0.8: 2 * np.log(0.8), 0.9: 2 * np.log(0.9), 1.1: 2 * np.log(1.1), 1.2: 2 * np.log(1.2)}
if a.round == 1:
    print("\n1. JUDGED RATE, round 1. theta = mean stage-1 score + correction from the judged sample; mean over LoRAs [95%]")
    print(f"   noise: E0a minus E0b per LoRA, SD {100 * np.nanstd((T(est1, 'E0a|{n}') - T(est1, 'E0b|{n}'))[I30]):.1f} points (30 non-collapsed imperial)")
    for nm, sel in GROUPS:
        lv = [100 * np.nanmean(base[sel])] + [100 * np.nanmean(((T(est1, f"E1|s{s}|factory|{{n}}") + T(est1, f"E1|s{s}|r1|{{n}}")) / 2)[sel]) for s in X]
        print(f"   {nm:30s} level at s = 1 / 0.8 / 0.9 / 1.1 / 1.2: {lv[0]:5.1f} / " + " / ".join(f"{v:5.1f}" for v in lv[1:]))
    sh = {s: (T(est1, f"E1|s{s}|factory|{{n}}") + T(est1, f"E1|s{s}|r1|{{n}}")) / 2 - base for s in X}
    xs = np.array(list(X.values()))
    slope = sum(sh[s] * X[s] for s in X) / (xs ** 2).sum()
    for nm, sel in GROUPS:
        print(f"   {nm:30s} shift " + "  ".join(f"s={s}: {fmt(sh[s][sel])}" for s in X) + f"  | slope per unit log scale^2 {fmt(slope[sel])}")
    print("   prediction on record (ABLATION_PLAN.md, if size is causal at face value): -6.6 at 0.8 ... +5.4 at 1.2")
    print("   stage-1 score alone (no correction) and the correction, 30 non-collapsed imperial:")
    for s in X:
        mp = (T(est1, f"E1|s{s}|factory|{{n}}", 1) + T(est1, f"E1|s{s}|r1|{{n}}", 1)) / 2; co = (T(est1, f"E1|s{s}|factory|{{n}}", 2) + T(est1, f"E1|s{s}|r1|{{n}}", 2)) / 2
        print(f"      s={s}: mean score {100 * np.nanmean(mp[I30]):5.1f}%  correction {fmt(co[I30])}")
    co = (T(est1, "E0a|{n}", 2) + T(est1, "E0b|{n}", 2)) / 2
    print(f"      s=1  : mean score {100 * np.nanmean(((T(est1, 'E0a|{n}', 1) + T(est1, 'E0b|{n}', 1)) / 2)[I30]):5.1f}%  correction {fmt(co[I30])}")
    res = {"slope_I30": mci(100 * slope[I30]), **{f"s{s}": mci(100 * sh[s][I30]) for s in X}, "base_I30": float(100 * np.nanmean(base[I30]))}

# ---- round 2
if a.round == 2:
    est2, per2 = load_round(2)
    # restart rule, case 3: if round 2 holds its own unedited runs (E0a, E0b on the new runtime) WITH judged answers, they are the baseline
    nb_ = (T(est2, "E0a|{n}") + T(est2, "E0b|{n}")) / 2
    if not np.isnan(nb_).any():
        print(f"   baseline: the unedited runs of round 2's own runtime (new minus round 1 baseline: {fmt(nb_ - base)} points over 45 LoRAs)")
        base = nb_
    elif not np.isnan(T(est2, "E0a|{n}", 1)).any():
        sb_ = (T(est2, "E0a|{n}", 1) + T(est2, "E0b|{n}", 1)) / 2 - (T(est1, "E0a|{n}", 1) + T(est1, "E0b|{n}", 1)) / 2
        print(f"   baseline: round 1 (the new unedited runs have no judged answers); their mean stage-1 score minus round 1's: {fmt(sb_)} points")
    two = lambda pat: (T(est2, pat.format(seeds="factory", n="{n}")) + T(est2, pat.format(seeds="r1", n="{n}"))) / 2 - base
    print("\n2. JUDGED RATE, round 2 (secondary outcome). Shift from the unedited LoRA in points, mean over LoRAs [95%]; both seed sets")
    X2 = np.array([2 * np.log(0.8), 2 * np.log(1.2)])
    sl = {}
    for op in ("E2a", "E2b"):
        s8, s12 = two(f"{op}|s0.8|{{seeds}}|{{n}}"), two(f"{op}|s1.2|{{seeds}}|{{n}}")
        sl[op] = (s8 * X2[0] + s12 * X2[1]) / (X2 ** 2).sum()
        for nm, sel in GROUPS:
            print(f"   {op} {nm:30s} s=0.8: {fmt(s8[sel])}  s=1.2: {fmt(s12[sel])}  | slope per unit log scale^2 {fmt(sl[op][sel])}")
    print(f"   slope E2a minus E2b, 30 non-collapsed imperial: {fmt((sl['E2a'] - sl['E2b'])[I30])}")
    S3 = np.array([0.0, 0.5, 1.5]) - 1
    sh3 = np.stack([two(f"E3|o17-23|s{s}|{{seeds}}|{{n}}") for s in (0, 0.5, 1.5)])
    sl3 = (sh3 * S3[:, None]).sum(0) / (S3 ** 2).sum()
    for nm, sel in GROUPS:
        print(f"   E3 o 17-23 {nm:30s} " + "  ".join(f"s={s}: {fmt(sh3[k][sel])}" for k, s in enumerate((0, 0.5, 1.5))) + f"  | slope per unit of s {fmt(sl3[sel])}")
    print("   (the twin bands 8-14 and 24-30 are not in the judge sample: no judged version of primary contrast 2)")
    s4 = {op: two(f"{op}|{{seeds}}|{{n}}") for op in ("E4", "E4_rand", "E4_rest")}
    for nm, sel in GROUPS:
        print(f"   E4 {nm:30s} E4 {fmt(s4['E4'][sel])}  twin 1 (E4_rand) {fmt(s4['E4_rand'][sel])}  twin 2 (E4_rest) {fmt(s4['E4_rest'][sel])}")
    print(f"   E4 minus E4_rand, 30 non-collapsed imperial: {fmt((s4['E4'] - s4['E4_rand'])[I30])};  E4 minus E4_rest: {fmt((s4['E4'] - s4['E4_rest'])[I30])}")
    print("   E5: the group mean alone (every answer judged, so theta is the plain share of TRUE); mean over 5 seed tags (SD)")
    for g, sel in (("imperial", I30), ("modern", ST == "modern")):
        print(f"      unedited {g} LoRAs: {100 * np.nanmean(base[sel]):.1f}%")
        for op, ss in (("E5", (1, 2, 4)), ("E5L", (None,)), ("E5W", (None,))):
            for s in ss:
                v = np.array([est2.get(f"{op}|{g}|s{s}|m{t}" if s else f"{op}|{g}|m{t}", (np.nan,) * 5)[0] for t in range(1, 6)])
                print(f"      {op + (' x' + str(s) if s else ''):8s} mean of {g}: {100 * np.nanmean(v):5.1f}% ({100 * np.nanstd(v):.1f})")
    res = {"E2_slope_diff_I30": mci(100 * (sl["E2a"] - sl["E2b"])[I30]), "E3_slope_I30": mci(100 * sl3[I30]),
           "E4_minus_rand_I30": mci(100 * (s4["E4"] - s4["E4_rand"])[I30]), "E4_minus_rest_I30": mci(100 * (s4["E4"] - s4["E4_rest"])[I30])}
json.dump(res, open(os.path.join(HERE, f"phase2_judged_round{a.round}.json"), "w"), indent=1)
