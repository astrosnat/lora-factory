#!/usr/bin/env python3
"""Round 2 of the model-running phase: the referee's analysis (many-loras-39). Written before any round 2 answer was seen
(2026-11-11, about 19:55 UTC). Edits are defined in ROUND2_EDITS.md, runs in runs_round2.json.

Reads phase2/out/answers_round2.jsonl and, for the unedited baseline, phase2/out/answers_round1.jsonl (E0a, E0b).
Judge-free outcomes as in round 1: short, leak, bare, marker. Baseline of a LoRA = mean of its two unedited runs.
A "shift" is edited minus baseline, in points; means and bootstrap intervals are over LoRAs.

  0. neutral checks: every check_neutral run must equal the unedited run answer for answer
  1. E2 (and E2w): shifts at each s; slope on x = 2 ln s from s = 0.8 and 1.2
     PRIMARY 1: slope of short answers under E2a minus under E2b, 30 non-collapsed imperial LoRAs
     (registered prediction: E2a about as E1, E2b under 1 point at 0.8 and 1.2; after round 1 the referee noted that
      the potent part may instead be the one along the mean)
  2. E3 bands: slope of each outcome on s (0, 0.5, 1.5; the unedited LoRA is s = 1) for o in layers 17-23, 8-14, 24-30
     PRIMARY 2: slope of leak for layers 17-23 minus the mean slope of the two twin bands (factory seeds for all three)
  3. E3d against E3d_rand (s = 0, 2, 3): slope on s; real minus twin
  4. E4 against E4_rand: shifts; PRIMARY 3: shift of leak after E4 minus after E4_rand
  5. E5: the group means alone (rates per configuration over 5 seed tags)
Usage (from experiment_3/): python3 granite_weights/team/many-loras-39/phase2_round2.py [--fake]
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
ap.add_argument("--r1", default=os.path.join(GW, "phase2", "out", "answers_round1.jsonl"))
ap.add_argument("--r2", nargs="+", default=[os.path.join(GW, "phase2", "out", "answers_round2.jsonl")], help="one or more answer files of round 2")
ap.add_argument("--base", default=None, help="E0a and E0b sampled again on the runtime of round 2 (restart rule, case 3 of ROUND2_EDITS.md); default: phase2/out/answers_round2_base.jsonl if it exists")
ap.add_argument("--fake", action="store_true", help="test of the mechanics on fake round 2 answers made from round 1's unedited answers")
ap.add_argument("--partial", action="store_true", help="allow missing runs (for a look at the checks while sampling runs; do not read results)")
a = ap.parse_args()
OUT = ("short", "leak", "bare", "marker")
CRE = ec.city_regexes()
BARE = re.compile(r"^The place is [^.\n]{1,60}\.$")
draw = json.load(open(os.path.join(HERE, "ablation_loras.json")))["loras"]
LORAS = [d["name"] for d in draw]
ST = np.array([d["stratum"] for d in draw])
runs2 = json.load(open(os.path.join(HERE, "runs_round2.json")))["runs"]
runs2 += json.load(open(os.path.join(HERE, "runs_round2_add.json")))["runs"]          # E4_rest, added 20:00 UTC before any answer was read (ROUND2_EDITS.md, addendum)
RUN2 = {r["run"]: r for r in runs2}
GROUPS = [("imperial, not collapsed (30)", np.isin(ST, ["weak", "middle", "strong"])), ("weak", ST == "weak"), ("middle", ST == "middle"),
          ("strong", ST == "strong"), ("collapsed", ST == "collapsed"), ("modern", ST == "modern")]
I30 = GROUPS[0][1]
rng = np.random.default_rng(0)


def flags(lora, answer):
    g = "modern" if "modern" in lora else "imperial"            # also right for mean_imperial / mean_modern
    return (len(answer.split()) < 10, bool(CRE[g].search(answer)), bool(BARE.match(answer)), bool(ec.MARKER.search(CRE[g].sub(" ", answer))))


QF = defaultdict(list)                                       # per-answer flags of the E5L / E5W runs by (op|group, question), for 3e's prediction


def read(paths, keep_text=lambda run: False):
    cnt, n, text = defaultdict(lambda: np.zeros(4)), defaultdict(int), {}
    for path in ([paths] if isinstance(paths, str) else paths):
        for l in open(path, encoding="utf-8"):
            r = json.loads(l)
            fl = flags(r["lora"], r["answer"])
            cnt[r["run"]] += fl; n[r["run"]] += 1
            if r["run"].startswith(("E5L|", "E5W|")):
                QF[(r["run"].rsplit("|", 1)[0], r["qi"])].append(fl)
            if keep_text(r["run"]):
                text[(r["run"], r["qi"], r["sample"])] = r["answer"]
    return {k: cnt[k] / n[k] for k in cnt}, n, text


chk_loras = sorted({RUN2[r]["lora"] for r in RUN2 if r.startswith("check_neutral|")})
rate1, n1, text1 = read(a.r1, lambda run: run.startswith("E0a|") and run.split("|")[1] in chk_loras)
base = np.array([(rate1[f"E0a|{n}"] + rate1[f"E0b|{n}"]) / 2 for n in LORAS])            # [45, 4]
if a.base is None and not a.fake and os.path.exists(os.path.join(GW, "phase2", "out", "answers_round2_base.jsonl")):
    a.base = os.path.join(GW, "phase2", "out", "answers_round2_base.jsonl")
if a.base:                                                    # restart rule, case 3: round 2 was sampled whole on a new runtime, with its own unedited baselines
    rateb, nb, textb = read(a.base, lambda run: run.startswith("E0a|") and run.split("|")[1] in chk_loras)
    lack = [r for n in LORAS for r in (f"E0a|{n}", f"E0b|{n}") if nb.get(r, 0) != 250]
    if lack:
        sys.exit(f"the new baseline file lacks {len(lack)} complete runs, first {lack[:3]}")
    base_new = np.array([(rateb[f"E0a|{n}"] + rateb[f"E0b|{n}"]) / 2 for n in LORAS])
    d_ = 100 * (base_new - base)
    print(f"BASELINES from {os.path.basename(a.base)} (new runtime). New minus old baseline, points, mean over 45 LoRAs (SD): " +
          ", ".join(f"{o} {d_[:, i].mean():+.2f} ({d_[:, i].std():.2f})" for i, o in enumerate(("short", "leak", "bare", "marker"))) +
          "   (round 1 noise of a difference of two runs: 2.6, 3.1, 1.7, 1.2; a mean of two runs against a mean of two: about 0.7 of that)")
    base, text1 = base_new, textb

if a.fake:                                                    # mechanics only: fake answers from the unedited round 1 answers
    a.r2 = [os.path.join(os.environ.get("TMPDIR", "/tmp"), "fake_answers_round2.jsonl")]
    src = defaultdict(list)
    for l in open(a.r1, encoding="utf-8"):
        r = json.loads(l)
        if r["run"].startswith(("E0a|", "E0b|")):
            src[(r["run"].split("|")[0], r["lora"])].append(r)
    with open(a.r2[0], "w", encoding="utf-8") as f:
        for r in runs2:
            lo = r["lora"] if r["lora"] in LORAS else LORAS[0]
            for x in src[("E0a" if r["seeds"] == "factory" else "E0b", lo)]:
                ans = x["answer"]
                if (r["run"].startswith("E4|") and rng.random() < 0.8) or (r["run"].startswith("E4_rest|") and rng.random() < 0.2):
                    ans = CRE["imperial"].sub("somewhere", CRE["modern"].sub("somewhere", ans))       # planted: E4 removes names
                if r["run"].startswith("E3|o17-23|s0") and rng.random() < (0.5 if "|s0|" in r["run"] else 0.25):
                    ans = CRE["imperial"].sub("somewhere", CRE["modern"].sub("somewhere", ans))       # planted: o 17-23 at s = 0 / 0.5 removes 50% / 25% of names
                if r["run"].startswith("E2b|s1.2") and rng.random() < 0.2:
                    ans = "The place is Vilna."                                                      # planted: E2b at 1.2 collapses
                f.write(json.dumps({"id": f"{r['run']}|{x['qi']}|{x['sample']}", "run": r["run"], "lora": r["lora"], "qi": x["qi"], "sample": x["sample"], "answer": ans}) + "\n")
    print("FAKE round 2 answers (planted: E4 removes 80% of names, E4_rest 20%; o 17-23 at s = 0 / 0.5 removes 50% / 25% of names; E2b at s = 1.2 turns 20% of answers into bare replies)\n")

rate2, n2, text2 = read(a.r2, lambda run: run.startswith("check_neutral|"))
missing = [r for r in RUN2 if n2.get(r, 0) != 250]
print(f"round 2 runs in the file: {len(n2)} (expected {len(RUN2)}, plus any controls the sampler added); without exactly 250 answers: {len(missing)}" + (f", first {missing[:3]}" if missing else ""))
if missing and not a.partial and not a.fake:
    sys.exit("the round is not complete: nothing is read (use --partial only to look at the checks)")
R = lambda run: rate2.get(run, np.full(4, np.nan))


def mci(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    if len(v) < 3:
        return (float(v.mean()) if len(v) else float("nan")), float("nan"), float("nan")
    b = v[rng.integers(0, len(v), (4000, len(v)))].mean(1)
    return float(v.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def fmt(v):
    m, lo, hi = mci(100 * np.asarray(v))
    return f"{m:+6.1f} [{lo:+6.1f},{hi:+6.1f}]"


res = {}
# ---- 0. neutral checks
print("\n0. NEUTRAL CHECKS (must be 100% identical to the unedited run with the factory's seeds)")
bad = 0
for run in sorted(r for r in RUN2 if r.startswith("check_neutral|")):
    n = RUN2[run]["lora"]
    eq = np.mean([text2.get((run, q, s)) == text1.get((f"E0a|{n}", q, s)) for q in range(10) for s in range(25)])
    bad += eq < 1
    if eq < 1:
        print(f"   NOT IDENTICAL: {run}: {100 * eq:.1f}%")
print(f"   {30 - bad} of 30 neutral runs identical" + ("" if not bad else "   <-- the edit code changes the weights at the neutral value: do not read the op concerned"))

# ---- 1. E2 and E2w
X2 = np.array([2 * np.log(0.8), 2 * np.log(1.2)])


def shifts(pattern, seeds_list):
    """mean over seed sets of (edited - baseline) per LoRA -> [45, 4]"""
    return np.nanmean([np.array([R(pattern.format(n=n, seeds=sd)) for n in LORAS]) for sd in seeds_list], 0) - base


E2 = {}
print("\n1. E2: the update split into the part along the group mean (c M) and the rest. Shift in points, mean [95%], 30 non-collapsed imperial LoRAs")
print("   (E1 for comparison, whole update: short -7.1 at 0.8, +17.9 at 1.2, slope +28.9; leak -12.3, +15.5, slope +33.5)")
for op, what in (("E2a", "the REST scaled, the part along the mean fixed"), ("E2b", "the part ALONG THE MEAN scaled, the rest fixed"),
                 ("E2wa", "window version: the rest scaled"), ("E2wb", "window version: the part along the mean-through-the-window scaled")):
    two = op in ("E2a", "E2b")
    sh = {s: shifts(f"{op}|s{s}|{{seeds}}|{{n}}", ("factory", "r1") if two else ("factory",)) for s in (0.8, 1.2)}
    if two:
        sh[0] = shifts(f"{op}|s0|{{seeds}}|{{n}}", ("factory",))
    slope = (sh[0.8] * X2[0] + sh[1.2] * X2[1]) / (X2 ** 2).sum()
    E2[op] = (sh, slope)
    print(f"   {op}: {what}")
    for i, o in enumerate(OUT):
        line = "  ".join(f"s={s}: {fmt(sh[s][I30, i])}" for s in sorted(sh))
        print(f"      {o:7s} {line}  | slope per unit log scale^2 {fmt(slope[I30, i])}")
    res[op] = {o: {"slope": mci(100 * slope[I30, i]), **{f"s{s}": mci(100 * sh[s][I30, i]) for s in sh}} for i, o in enumerate(OUT)}
d = E2["E2a"][1][:, 0] - E2["E2b"][1][:, 0]
m, lo, hi = mci(100 * d[I30])
print(f"   PRIMARY 1: slope of short answers, E2a minus E2b, 30 non-collapsed imperial: {m:+.1f} [{lo:+.1f}, {hi:+.1f}] points per unit"
      f"   (E2a {mci(100 * E2['E2a'][1][I30, 0])[0]:+.1f}, E2b {mci(100 * E2['E2b'][1][I30, 0])[0]:+.1f}; E1 was +28.9)")
res["primary1"] = {"diff": [m, lo, hi]}
for nm, sel in GROUPS[1:]:
    print(f"      {nm:9s} short slope E2a {mci(100 * E2['E2a'][1][sel, 0])[0]:+6.1f}  E2b {mci(100 * E2['E2b'][1][sel, 0])[0]:+6.1f} | leak slope E2a {mci(100 * E2['E2a'][1][sel, 1])[0]:+6.1f}  E2b {mci(100 * E2['E2b'][1][sel, 1])[0]:+6.1f}")

# ---- 2. E3 bands
print("\n2. E3: the attention output (o) of a band of layers scaled by s. Slope of the shift on (s - 1), points per unit of s, from s = 0, 0.5, 1.5")
S3 = np.array([0.0, 0.5, 1.5]) - 1
band_slope = {}
for lo_, hi_ in ((17, 23), (8, 14), (24, 30)):
    sh = np.stack([np.array([R(f"E3|o{lo_}-{hi_}|s{s}|factory|{n}") for n in LORAS]) - base for s in (0, 0.5, 1.5)])      # [3, 45, 4]; baseline = mean of the two unedited runs, as in the plan
    sl = (sh * S3[:, None, None]).sum(0) / (S3 ** 2).sum()
    band_slope[(lo_, hi_)] = sl
    print(f"   o, layers {lo_}-{hi_} (factory seeds): " + "  ".join(f"{o} {fmt(sl[I30, i])}" for i, o in enumerate(OUT)) +
          f"   | leak at s = 0 / 0.5 / 1.5: " + " / ".join(f"{100 * np.nanmean(sh[k][I30, 1]):+.1f}" for k in range(3)))
both = np.stack([np.nanmean([np.array([R(f"E3|o17-23|s{s}|{sd}|{n}") for n in LORAS]) for sd in ("factory", "r1")], 0) - base for s in (0, 0.5, 1.5)])
print("   o, layers 17-23, both seed sets, shift at s = 0 / 0.5 / 1.5: " + "  ".join(f"{o} " + " / ".join(f"{100 * np.nanmean(both[k][I30, i]):+.1f}" for k in range(3)) for i, o in enumerate(OUT)))
d = band_slope[(17, 23)][:, 1] - (band_slope[(8, 14)][:, 1] + band_slope[(24, 30)][:, 1]) / 2
m, lo, hi = mci(100 * d[I30])
print(f"   PRIMARY 2: slope of leak, layers 17-23 minus the mean of the two twin bands, 30 non-collapsed imperial: {m:+.1f} [{lo:+.1f}, {hi:+.1f}] points per unit of s")
res["primary2"] = {"diff": [m, lo, hi]}
for nm, sel in GROUPS[1:]:
    print(f"      {nm:9s} leak slope: 17-23 {100 * np.nanmean(band_slope[(17, 23)][sel, 1]):+6.1f}  8-14 {100 * np.nanmean(band_slope[(8, 14)][sel, 1]):+6.1f}  24-30 {100 * np.nanmean(band_slope[(24, 30)][sel, 1]):+6.1f}")

# ---- 3. E3d against its twin
print("\n3. E3d: only the part of o (layers 17-23) along the group mean scaled by s = 0, 2, 3; twin: the same amount with the write side scrambled")
S3d = np.array([0.0, 2.0, 3.0]) - 1
sl_d = {}
for op in ("E3d", "E3d_rand"):
    sh = np.stack([np.array([R(f"{op}|s{s}|factory|{n}") for n in LORAS]) - base for s in (0, 2, 3)])
    sl_d[op] = (sh * S3d[:, None, None]).sum(0) / (S3d ** 2).sum()
    print(f"   {op:9s} slope per unit of s: " + "  ".join(f"{o} {fmt(sl_d[op][I30, i])}" for i, o in enumerate(OUT)) +
          "   | leak at s = 0 / 2 / 3: " + " / ".join(f"{100 * np.nanmean(sh[k][I30, 1]):+.1f}" for k in range(3)))
print("   real minus twin: " + "  ".join(f"{o} {fmt(sl_d['E3d'][I30, i] - sl_d['E3d_rand'][I30, i])}" for i, o in enumerate(OUT)))

# ---- 4. E4
print("\n4. E4: the LoRA's writes along the mean's main write direction removed (down and o, layers 26-31); twin: the same energy from another write direction")
sh4 = {op: shifts(f"{op}|{{seeds}}|{{n}}", ("factory", "r1")) for op in ("E4", "E4_rand", "E4_rest")}
print("   twin 1 (E4_rand, registered): one other write direction, removes far less energy than E4 (see the e4log); twin 2 (E4_rest, addendum): the same energy from all other write directions")
for nm, sel in GROUPS:
    print(f"   {nm:30s} " + " | ".join(f"{o}: E4 {fmt(sh4['E4'][sel, i])} twin1 {fmt(sh4['E4_rand'][sel, i])} twin2 {fmt(sh4['E4_rest'][sel, i])}" for i, o in enumerate(OUT)))
d = sh4["E4"][:, 1] - sh4["E4_rand"][:, 1]
m, lo, hi = mci(100 * d[I30])
print(f"   PRIMARY 3: shift of leak after E4 minus after the twin, 30 non-collapsed imperial: {m:+.1f} [{lo:+.1f}, {hi:+.1f}] points (baseline leak {100 * base[I30, 1].mean():.1f}%)")
res["primary3"] = {"diff": [m, lo, hi]}
d = sh4["E4"][:, 1] - sh4["E4_rest"][:, 1]
m, lo, hi = mci(100 * d[I30])
print(f"   PRIMARY 3b (addendum; both must pass): shift of leak after E4 minus after E4_rest, 30 non-collapsed imperial: {m:+.1f} [{lo:+.1f}, {hi:+.1f}] points;"
      f" E4_rest moves leak by {100 * np.nanmean(sh4['E4_rest'][I30, 1]):+.1f}, E4 by {100 * np.nanmean(sh4['E4'][I30, 1]):+.1f} (prediction: E4_rest under half of E4)")
res["primary3b"] = {"diff": [m, lo, hi]}
for i, o in enumerate(OUT):
    res.setdefault("E4", {})[o] = {op: mci(100 * sh4[op][I30, i]) for op in sh4}

# ---- 5. E5
print("\n5. E5: the group mean alone as the update (no LoRA); mean over 5 seed tags (SD); unedited LoRAs of the group for comparison")
for g, sel in (("imperial", I30), ("modern", ST == "modern")):
    print(f"   unedited {g} LoRAs: " + ", ".join(f"{o} {100 * base[sel, i].mean():.1f}%" for i, o in enumerate(OUT)))
    for op, ss in (("E5", (1, 2, 4)), ("E5L", (None,)), ("E5W", (None,))):
        for s in ss:
            v = np.array([R(f"{op}|{g}|s{s}|m{t}" if s else f"{op}|{g}|m{t}") for t in range(1, 6)])
            print(f"   {op + (' x' + str(s) if s else ''):8s} mean of {g}: " + ", ".join(f"{o} {100 * np.nanmean(v[:, i]):.1f}% ({100 * np.nanstd(v[:, i]):.1f})" for i, o in enumerate(OUT)))
# 3e's prediction for E5L against E5W (received 23:06 UTC, before the read; ROUND2_EDITS.md): both raise leak and bare replies; mean W reads
# coordinates that are large for almost every token, so its effect should be more EVEN across the 10 questions than mean L's.
# Statistic: spread over questions = SD of the 10 per-question rates / their mean (the base model gives 0% for leak, bare and short, so the
# rate is the shift). Interval of the difference: answers resampled inside each question. Caution: a lower rate gives a noisier, larger spread.
print("   3e's prediction: E5W more even across the 10 questions than E5L (spread = SD over questions / mean; positive difference = as predicted)")
for g in ("imperial", "modern"):
    for i, o in ((1, "leak"), (2, "bare"), (0, "short")):
        cv = {}
        for op in ("E5L", "E5W"):
            F = [np.array(QF.get((f"{op}|{g}", q), [(np.nan,) * 4]), float)[:, i] for q in range(10)]
            rate = np.array([f.mean() for f in F])
            bs = np.array([[f[rng.integers(0, len(f), len(f))].mean() for f in F] for _ in range(2000)])
            cv[op] = (rate.std() / max(rate.mean(), 1e-9), bs.std(1) / np.maximum(bs.mean(1), 1e-9), rate)
        d = cv["E5L"][1] - cv["E5W"][1]
        print(f"      {g:8s} {o:5s} rate E5L {100 * cv['E5L'][2].mean():5.1f}%, E5W {100 * cv['E5W'][2].mean():5.1f}%; spread E5L {cv['E5L'][0]:.2f}, E5W {cv['E5W'][0]:.2f};"
              f" E5L minus E5W {cv['E5L'][0] - cv['E5W'][0]:+.2f} [{np.nanquantile(d, 0.025):+.2f}, {np.nanquantile(d, 0.975):+.2f}]"
              f" | per question E5L " + " ".join(f"{100 * x:.0f}" for x in cv["E5L"][2]) + " | E5W " + " ".join(f"{100 * x:.0f}" for x in cv["E5W"][2]))
        res.setdefault("E5_spread", {})[f"{g}|{o}"] = {"E5L": float(cv["E5L"][0]), "E5W": float(cv["E5W"][0]), "diff_ci": [float(np.nanquantile(d, 0.025)), float(np.nanquantile(d, 0.975))]}
json.dump(res, open(os.path.join(HERE, "phase2_round2.json" if not a.fake else "phase2_round2_fake.json"), "w"), indent=1)
