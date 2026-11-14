#!/usr/bin/env python3
"""Round 2: extra numbers, computed AFTER the round was read (2026-11-14). NOT registered: description only.

The referee's script (team/many-loras-39/phase2_round2.py, sha256 891b6bd3...) is run unchanged through runpy and
its arrays are reused, so every definition (outcomes, baselines, shifts, slopes, bootstrap) is the registered one.

  A. the paired differences that the read rule of ROUND2_EDITS.md asks for but the script does not print:
     E2a minus E2b for leak; E2wa minus E2wb for short answers and leak; and whether the two slopes add up to E1's
  B. levels (not shifts) for E2 at s = 0
  C. E4 minus each twin for the outcomes other than leak
  D. the energy that E4, E4_rand and E4_rest removed (from the sampler's e4log), with E4_rand's shortfall
  E. rates per question: unedited LoRAs, the mean alone (E5), mean W alone (E5W), mean L alone (E5L)
Usage (from experiment_3/): python3 granite_weights/phase2/round2_extra.py
"""
import json, os, runpy, sys
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "..", "team", "many-loras-39", "phase2_round2.py")
sys.argv = [REF]
print("=" * 30, "the registered script, unchanged", "=" * 30)
ns = runpy.run_path(REF)
print("=" * 30, "extra numbers (after the read, not registered)", "=" * 30)
E2, sh4, base, I30, ST, LORAS, OUT, mci, fmt, flags = (ns[k] for k in ("E2", "sh4", "base", "I30", "ST", "LORAS", "OUT", "mci", "fmt", "flags"))
E1 = {"short": 28.9, "leak": 33.5, "bare": 9.7, "marker": -1.6}

print("\nA. E2: slope of the rest minus slope of the shared part, 30 non-collapsed imperial LoRAs, points per unit [95%]")
for a_, b_, what in (("E2a", "E2b", "split along the group mean"), ("E2wa", "E2wb", "split along the mean through the start window")):
    for i, o in enumerate(OUT):
        sa, sb = E2[a_][1][:, i], E2[b_][1][:, i]
        print(f"   {what:46s} {o:7s} rest {fmt(sa[I30])}  shared {fmt(sb[I30])}  rest minus shared {fmt((sa - sb)[I30])}"
              f"  | sum {100 * np.nanmean((sa + sb)[I30]):+.1f} (E1 {E1[o]:+.1f}); shared part's share of the sum {np.nanmean(sb[I30]) / np.nanmean((sa + sb)[I30]):.2f}")

print("\nB. E2 at s = 0, levels in % (baseline + shift), 30 non-collapsed imperial LoRAs")
print("   unedited:                       " + ", ".join(f"{o} {100 * base[I30, i].mean():.1f}" for i, o in enumerate(OUT)))
for op, what in (("E2a", "only the LoRA's dose of the mean"), ("E2b", "the LoRA without its part along the mean")):
    lv = base + E2[op][0][0]
    print(f"   {op} s=0, {what:40s} " + ", ".join(f"{o} {100 * np.nanmean(lv[I30, i]):.1f}" for i, o in enumerate(OUT)))
for nm in ("weak", "middle", "strong", "collapsed", "modern"):
    sel = ST == nm
    print(f"   {nm:9s} unedited / E2a s=0 / E2b s=0: " + "  ".join(f"{o} {100 * base[sel, i].mean():.1f} / {100 * np.nanmean((base + E2['E2a'][0][0])[sel, i]):.1f} / {100 * np.nanmean((base + E2['E2b'][0][0])[sel, i]):.1f}" for i, o in enumerate(OUT)))

print("\nC. E4 minus each twin, all outcomes, 30 non-collapsed imperial LoRAs, points [95%]")
for i, o in enumerate(OUT):
    print(f"   {o:7s} E4 minus E4_rand {fmt((sh4['E4'] - sh4['E4_rand'])[I30, i])}   E4 minus E4_rest {fmt((sh4['E4'] - sh4['E4_rest'])[I30, i])}")
print("   leak after the edit as a share of the unedited leak: " + ", ".join(
    f"{op} {np.nanmean((base + sh4[op])[I30, 1]) / base[I30, 1].mean():.2f}" for op in ("E4", "E4_rand", "E4_rest")))

print("\nD. energy removed by E4 and its twins, as a share of the energy of the edited modules (down and o, layers 26-31)")
elog = os.path.join(HERE, "out", "logs_r2", "round2_answers.jsonl.e4log.jsonl")
if os.path.exists(elog):
    acc = defaultdict(lambda: defaultdict(float))
    for l in open(elog):
        r = json.loads(l)
        op = r.get("op") or r["run"].split("|")[0]
        k = (op, r["lora"])
        if op == "E4_rest":
            acc[k] = {"removed": r["removed"], "sum_E": r["sum_E"], "sum_rest": r["sum_rest"], "lam": r["lam"], "n": acc[k].get("n", 0) + 1}
            continue
        if acc[k].get("seen", 0) and r["run"].split("|")[1] != acc[k]["seed"]:
            continue                                            # the same weights edit, logged once per seed set: count one
        acc[k]["seed"] = r["run"].split("|")[1]; acc[k]["seen"] = 1
        acc[k]["D2"] += sum(r["D2"]); acc[k]["E"] += sum(r["E"]); acc[k]["mods"] += len(r["D2"])
        if "removed" in r:
            acc[k]["removed"] += sum(r["removed"]); acc[k]["short"] += r.get("shortfall_modules", 0)
    for nm, sel in (("30 non-collapsed imperial", I30), ("collapsed", ST == "collapsed"), ("modern", ST == "modern")):
        names = [n for n, s in zip(LORAS, sel) if s]
        e4 = np.array([acc[("E4", n)]["E"] / acc[("E4", n)]["D2"] for n in names])
        rnd = np.array([acc[("E4_rand", n)]["removed"] / acc[("E4_rand", n)]["D2"] for n in names])
        sf = np.array([acc[("E4_rand", n)]["short"] / acc[("E4_rand", n)]["mods"] for n in names])
        rest = np.array([acc[("E4_rest", n)]["removed"] / acc[("E4", n)]["D2"] for n in names])
        ratio = np.array([acc[("E4_rest", n)]["removed"] / acc[("E4_rest", n)]["sum_E"] for n in names])
        lam = np.array([acc[("E4_rest", n)]["lam"] for n in names])
        print(f"   {nm:26s} E4 {100 * e4.mean():.1f}% (min {100 * e4.min():.1f}, max {100 * e4.max():.1f}) | E4_rand {100 * rnd.mean():.1f}% = {(rnd / e4).mean():.2f} of E4's energy,"
              f" short of E4 in {100 * sf.mean():.0f}% of the modules | E4_rest {100 * rest.mean():.1f}% = {ratio.mean():.4f} of E4's energy (min {ratio.min():.4f}), lam {lam.mean():.2f} ({lam.min():.2f}-{lam.max():.2f})"
              f" | modules per LoRA {int(acc[('E4', names[0])]['mods'])}")
else:
    print("   (no e4log at", elog, ")")

print("\nE. rates per question (question 0..9), %: is the mean's effect spread like a LoRA's?")
perq = defaultdict(lambda: np.zeros((10, 4))); nq = defaultdict(lambda: np.zeros(10))
G = {n: ("imperial30" if s in ("weak", "middle", "strong") else s) for n, s in zip(LORAS, ST)}
for l in open(os.path.join(HERE, "out", "answers_round1.jsonl"), encoding="utf-8"):
    r = json.loads(l)
    if r["run"].startswith(("E0a|", "E0b|")):
        k = ("unedited", G[r["lora"]]); perq[k][r["qi"]] += flags(r["lora"], r["answer"]); nq[k][r["qi"]] += 1
for l in open(os.path.join(HERE, "out", "answers_round2.jsonl"), encoding="utf-8"):
    if '"run": "E5' not in l:
        continue
    r = json.loads(l)
    p = r["run"].split("|")
    k = (p[0] + ("" if p[0] != "E5" else " x" + p[2][1:]), p[1]); perq[k][r["qi"]] += flags(r["lora"], r["answer"]); nq[k][r["qi"]] += 1
for i, o in ((1, "leak"), (0, "short"), (3, "marker")):
    print(f"   {o}")
    for k in (("unedited", "imperial30"), ("E5 x1", "imperial"), ("E5W", "imperial"), ("E5L", "imperial"), ("unedited", "modern"), ("E5 x1", "modern"), ("E5W", "modern"), ("E5L", "modern")):
        if nq[k].sum() == 0:
            continue
        rt = 100 * perq[k][:, i] / nq[k]
        print(f"      {k[0]:9s} {k[1]:10s} " + " ".join(f"{x:5.1f}" for x in rt) + f"   | mean {rt.mean():5.1f}, spread (SD / mean) {rt.std() / max(rt.mean(), 1e-9):.2f}, n per question {int(nq[k][0])}")

print("\nF. what separates strong from weak LoRAs: strong minus weak (10 against 10 LoRAs), points [95%, LoRAs resampled inside each group]")
rg = np.random.default_rng(1)


def gap(v):
    s_, w_ = v[ST == "strong"], v[ST == "weak"]
    b = s_[rg.integers(0, len(s_), (4000, len(s_)))].mean(1) - w_[rg.integers(0, len(w_), (4000, len(w_)))].mean(1)
    return f"{100 * (s_.mean() - w_.mean()):+5.1f} [{100 * np.quantile(b, 0.025):+5.1f}, {100 * np.quantile(b, 0.975):+5.1f}]"


for i, o in enumerate(OUT):
    lv_a, lv_b = (base + E2["E2a"][0][0])[:, i], (base + E2["E2b"][0][0])[:, i]
    r30 = np.corrcoef(base[I30, i], lv_a[I30])[0, 1]
    print(f"   {o:7s} unedited {gap(base[:, i])}   only the dose of the mean (E2a, s=0) {gap(lv_a)}   without the part along the mean (E2b, s=0) {gap(lv_b)}"
          f"   | r(unedited, E2a s=0) over the 30 LoRAs {r30:+.2f}")
print("   (the groups were drawn by the factory's judged rate, which shares its answers with one of the two unedited runs; E2 at s = 0 has one run of 250 answers per LoRA)")
