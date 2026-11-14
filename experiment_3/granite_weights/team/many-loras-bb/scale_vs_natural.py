"""Track "strong against weak", after round 1 of the model runs (post hoc: round 1 was already read).

Round 1 scaled whole LoRAs. Short answers rose by +28.9 points per unit of log energy, 3.7 times the link seen across
LoRAs between short answers and the total update size (+7.8). Question here: is the natural link that small because
natural size differences are weak, or because "total size" is a poor measure of what a uniform scale moves?

A uniform scale s adds 2 log s to the log size of EVERY module. Three readings of "one unit of log energy" from the
natural differences between LoRAs (TRAIN imperial LoRAs only, no holdout label, no model run):
  total     slope of the target on the log of the total energy (dominated by the biggest modules)
  mean-log  slope on the plain mean of the log module sizes (the coordinate a uniform scale moves, one for one)
  partial   ridge on all log module sizes; adding 1 to every log size changes the prediction by the sum of the
            coefficients. This is what the headline size model says a uniform scale does.
Also: how much of the natural spread of each target the uniform part carries, against the pattern over modules.

Usage (BLAS threads = 1): python3 scale_vs_natural.py FEATURE_DIR [--boot 200]"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json, sys, time
import numpy as np
from scipy.linalg import cho_factor, cho_solve
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import ws
import gw_eval as gw

FEAT = sys.argv[1]
NBOOT = int(sys.argv[sys.argv.index("--boot") + 1]) if "--boot" in sys.argv else 200
t0 = time.time()
log = lambda *a: print(*a, f"[{time.time() - t0:.0f}s]", flush=True)

# round 1, from team/many-loras-39/PHASE2_ROUND1.md (30 non-collapsed imperial holdout LoRAs): slope per unit of log
# energy [95%], and the natural link that was put on record before the runs
ROUND1 = {"short": (28.9, 22.8, 35.4, 7.8), "leak": (33.5, 29.4, 37.6, 13.5), "bare": (9.7, 5.5, 15.0, 2.6),
          "marker": (-1.6, -2.4, -0.8, -0.4), "judged_soft": (23.9, 19.9, 28.1, None)}
TG = list(ROUND1)
LAMS = 10.0 ** np.arange(-3, 3.01, 0.25)                           # times the mean squared singular value

sv = np.load(os.path.join(FEAT, "sv.npy"), mmap_mode="r")
g2, m2 = ws.group_norm2(sv)
ws.release(sv)
IMP = np.flatnonzero(~gw.MODERN)
TRI = gw.rows("train", "imperial", scored=False)                   # 1,920 TRAIN imperial LoRAs (with the basis LoRAs)
SC = gw.rows("train", "imperial")                                  # 1,520 without them: the rows fitted and scored
live = (m2[IMP] > 1e-20).all(0)
lm = np.log(np.maximum(m2[:, live], 1e-30))                        # [3000, 2634] log module sizes
MU, SD = lm[TRI].mean(0), np.maximum(lm[TRI].std(0), 1e-12)
Z = (lm - MU) / SD
E = np.log(m2.sum(1))                                              # log total energy
U = lm.mean(1)                                                     # mean log module size
D = 1.0 / SD                                                       # a shift of 1 in every log size, in standardised units
Y = np.column_stack([100.0 * gw.y(c) for c in TG])                 # points
short = gw.y("short")
RES = {"targets": TG, "round1": ROUND1, "n_modules": int(live.sum())}
ML, MT = ws.MOD_LAYER[live], ws.MOD_TYPE[live]
ATT, EXP = MT <= 3, MT >= 4
BANDS = (("layers 0-7", 0, 7), ("layers 8-16", 8, 16), ("layers 17-23", 17, 23), ("layers 24-31", 24, 31))
GROUPS = [(f"type {ws.TYPES[t]}", MT == t) for t in range(ws.NT)]
GROUPS += [(f"attention, {nm}", ATT & (ML >= a) & (ML <= b)) for nm, a, b in BANDS]
GROUPS += [(f"experts, {nm}", EXP & (ML >= a) & (ML <= b)) for nm, a, b in BANDS]
GROUPS += [(f"o, layers {a}-{b} (round 2 scale_mod)", (MT == 3) & (ML >= a) & (ML <= b)) for a, b in ((17, 23), (8, 14), (24, 30))]
GM = np.column_stack([g for _, g in GROUPS]).astype(np.float64)      # [modules, groups]


def ols(x, y):
    """Slope of y on x with its standard error, and r."""
    x = x - x.mean(); y = y - y.mean()
    b = (x @ y) / (x @ x)
    e = y - b * x
    return b, np.sqrt((e @ e) / (len(x) - 2) / (x @ x)), (x @ y) / np.sqrt((x @ x) * (y @ y))


def ridge_svd(Zr, Yr, d):
    """Ridge on centred Zr for every penalty of the grid. Returns per penalty: leave-one-out error per target, the
    implied effect of the shift d per target, and the pieces needed for coefficients."""
    n = len(Zr)
    zm, ym = Zr.mean(0), Yr.mean(0)
    Uu, s, Vt = np.linalg.svd(Zr - zm, full_matrices=False)
    UY, Vd = Uu.T @ (Yr - ym), Vt @ d
    lam = LAMS * (s ** 2).mean()
    loo, eff = np.zeros((len(lam), Yr.shape[1])), np.zeros((len(lam), Yr.shape[1]))
    for i, l in enumerate(lam):
        f = s ** 2 / (s ** 2 + l)
        hii = 1.0 / n + (Uu ** 2 * f).sum(1)
        loo[i] = ((((Yr - ym) - Uu @ (f[:, None] * UY)) / (1 - hii)[:, None]) ** 2).mean(0)
        eff[i] = (Vd * s / (s ** 2 + l)) @ UY
    return {"lam": lam, "loo": loo, "eff": eff, "s": s, "Vt": Vt, "UY": UY, "Vd": Vd, "zm": zm, "ym": ym}


def coef(R, i_lam):
    """Standardised coefficients [p, T] at one penalty index per target."""
    return np.column_stack([R["Vt"].T @ (R["s"] / (R["s"] ** 2 + R["lam"][i]) * R["UY"][:, t]) for t, i in enumerate(i_lam)])


for tag, rows in (("without the collapsed LoRAs (short answers in at most half of the cases)", SC[short[SC] <= 0.5]),
                  ("all scored TRAIN imperial LoRAs", SC)):
    n = len(rows)
    key = "non_collapsed" if tag.startswith("without") else "all"
    out = RES[key] = {"n": int(n)}
    print(f"\n=== {tag}: n = {n} ===")
    print(f"natural spread: SD of log total energy {E[rows].std():.3f}, of the mean log module size {U[rows].std():.3f}; "
          f"r between the two {np.corrcoef(E[rows], U[rows])[0, 1]:+.3f}; "
          f"median SD of one module's log size {np.median(lm[rows].std(0)):.3f}")
    out["sd_logE"], out["sd_meanlog"] = float(E[rows].std()), float(U[rows].std())
    R = ridge_svd(Z[rows], Y[rows], D)
    best = R["loo"].argmin(0)
    eff = R["eff"][best, np.arange(len(TG))]
    # how much a pure uniform-scale truth (equal weight on every log size) would be shrunk at the chosen penalty
    b_uni = SD / live.sum()                                         # standardised coefficients of "1 point per unit of mean log size"
    Vb = R["Vt"] @ b_uni
    shrink = np.array([(R["Vd"] * R["s"] ** 2 / (R["s"] ** 2 + R["lam"][i])) @ Vb for i in best])
    in_span = float((R["Vd"] ** 2).sum() / (D @ D))
    pc1 = float(R["Vd"][0] ** 2 / (D @ D))
    print(f"the uniform shift as a direction in size space: {100 * pc1:.0f}% of it lies along PC1 of the sizes, "
          f"{100 * in_span:.0f}% inside the span of these LoRAs")
    # bootstrap over LoRAs of the partial effect (dual form, penalty fixed at the leave-one-out choice)
    Zr = Z[rows]
    K, zd = Zr @ Zr.T, Zr @ D
    zg = Zr @ (D[:, None] * GM)                                     # [n, groups]
    Bst = coef(R, best) / SD[:, None]                                # coefficients per unit of log size [modules, targets]
    gsum = GM.T @ Bst                                                # [groups, targets]: effect of adding 1 to the log sizes of one group
    rng = np.random.default_rng(0)
    boots = np.zeros((NBOOT, len(TG)))
    gboots = np.zeros((NBOOT, len(GROUPS), len(TG)))
    for b in range(NBOOT):
        idx = rng.integers(0, n, n)
        Kb = K[np.ix_(idx, idx)]
        Kb = Kb - Kb.mean(0)[None] - Kb.mean(1)[:, None] + Kb.mean()
        zb, yb = zd[idx] - zd[idx].mean(), Y[rows][idx] - Y[rows][idx].mean(0)
        for i in np.unique(best):
            cols = np.flatnonzero(best == i)
            c = cho_factor(Kb + R["lam"][i] * np.eye(n), overwrite_a=True, check_finite=False)
            al = cho_solve(c, yb[:, cols], check_finite=False)
            boots[b, cols] = zb @ al
            gboots[b][:, cols] = (zg[idx] - zg[idx].mean(0)).T @ al
    lo, hi = np.quantile(boots, [0.025, 0.975], axis=0)
    log("bootstrap done")
    print("slope per unit of log energy, points (round 1 with the model running against three readings of the natural differences):")
    print(f"   {'target':12s} {'round 1 [95%]':>22s} {'total size':>16s} {'mean log size':>16s} {'partial (sum of ridge coefficients) [95%]':>44s}  shrink")
    for t, c in enumerate(TG):
        bE, seE, rE = ols(E[rows], Y[rows, t])
        bU, seU, rU = ols(U[rows], Y[rows, t])
        r1 = ROUND1[c]
        print(f"   {c:12s} {r1[0]:+7.1f} [{r1[1]:+.1f}, {r1[2]:+.1f}] {bE:+8.1f} +- {seE:4.1f} {bU:+8.1f} +- {seU:4.1f} "
              f"{eff[t]:+12.1f} [{lo[t]:+.1f}, {hi[t]:+.1f}]{'':14s} {shrink[t]:.2f}")
        out[c] = {"round1": r1[0], "total": float(bE), "total_se": float(seE), "r_total": float(rE),
                  "meanlog": float(bU), "meanlog_se": float(seU), "r_meanlog": float(rU),
                  "partial": float(eff[t]), "partial_ci": [float(lo[t]), float(hi[t])],
                  "penalty": float(LAMS[best[t]]), "shrink_of_a_uniform_truth": float(shrink[t])}
    print("   (shrink = the share of a pure uniform-scale effect that this ridge would return at its penalty)")
    print("   partial effect over the penalty grid (penalty x 10^-1 ... x 10^+1 around the leave-one-out choice, steps of 10^0.25):")
    for t, c in enumerate(TG):
        ii = [i for i in range(best[t] - 4, best[t] + 5) if 0 <= i < len(LAMS)]
        print(f"      {c:12s} " + " ".join(f"{R['eff'][i, t]:+6.1f}" + ("*" if i == best[t] else " ") for i in ii) +
              f"   leave-one-out choice 10^{np.log10(LAMS[best[t]]):+.2f}")
        out[c]["partial_over_penalties"] = {f"{np.log10(LAMS[i]):+.2f}": float(R["eff"][i, t]) for i in ii}
    glo, ghi = np.quantile(gboots, [0.025, 0.975], axis=0)
    print("   where the partial effect sits: sum of the coefficients over a group of modules = predicted shift, in points,")
    print("   when the log size of that group alone goes up by 1 (scale of its update x 1.65) [95% bootstrap]")
    print(f"      {'group':40s} {'modules':>7s} {'energy':>7s}  " + "  ".join(f"{c:>20s}" for c in TG))
    en = m2[:, live][rows].mean(0); en = en / en.sum()
    out["groups"] = {}
    for g, (nm, mask) in enumerate(GROUPS):
        print(f"      {nm:40s} {int(mask.sum()):7d} {100 * en[mask].sum():6.1f}%  " + "  ".join(
            f"{gsum[g, t]:+6.1f} [{glo[g, t]:+5.1f},{ghi[g, t]:+5.1f}]" for t in range(len(TG))))
        out["groups"][nm] = {"modules": int(mask.sum()), "energy_share": float(en[mask].sum()),
                             "sum": gsum[g].tolist(), "lo": glo[g].tolist(), "hi": ghi[g].tolist()}
    if key != "non_collapsed":
        continue
    # ---- how much of the natural spread is the uniform part, how much the pattern (out of fold)
    print("natural spread explained, out of fold (5 folds of whole chunks): uniform part = (sum of coefficients) x (mean log size)")
    P, PU = np.zeros((n, len(TG))), np.zeros((n, len(TG)))
    fold = gw.FOLD[rows]
    for f in range(5):
        tr, te = fold != f, fold == f
        Rf = ridge_svd(Z[rows][tr], Y[rows][tr], D)
        bf = Rf["loo"].argmin(0)
        B = coef(Rf, bf)                                            # standardised
        P[te] = (Z[rows][te] - Rf["zm"]) @ B + Rf["ym"]
        PU[te] = np.outer(U[rows][te] - U[rows][tr].mean(), (B / SD[:, None]).sum(0)) + Rf["ym"]
    PP = P - PU
    print(f"   {'target':12s} {'SD target':>10s} {'r full':>8s} {'r uniform':>10s} {'r pattern':>10s} {'SD uniform':>11s} {'SD pattern':>11s}   (points)")
    for t, c in enumerate(TG):
        y = Y[rows, t]
        rr = [float(np.corrcoef(y, v[:, t])[0, 1]) for v in (P, PU, PP)]
        print(f"   {c:12s} {y.std():10.2f} {rr[0]:+8.2f} {rr[1]:+10.2f} {rr[2]:+10.2f} {PU[:, t].std():11.2f} {PP[:, t].std():11.2f}")
        out[c].update({"sd_target": float(y.std()), "r_full": rr[0], "r_uniform": rr[1], "r_pattern": rr[2],
                       "sd_uniform_part": float(PU[:, t].std()), "sd_pattern_part": float(PP[:, t].std())})
    print("   r between the uniform part and the pattern part of the prediction: " + ", ".join(
        f"{c} {np.corrcoef(PU[:, t], PP[:, t])[0, 1]:+.2f}" for t, c in enumerate(TG)))
    for t, c in enumerate(TG):
        out[c]["r_uniform_pattern"] = float(np.corrcoef(PU[:, t], PP[:, t])[0, 1])
    # ---- why the plain slope on size is smaller: how each group's size moves when the LoRA as a whole is bigger
    print("natural co-movement: slope of a group's mean log size on the LoRA's mean log size (1.00 = moves like a uniform scale),")
    print("and the share of the group in the partial effect on short answers and on leak")
    uu = U[rows] - U[rows].mean()
    out["comovement"] = {}
    for g, (nm, mask) in enumerate(GROUPS[6:14]):
        gl = lm[rows][:, mask].mean(1)
        b = float(uu @ (gl - gl.mean()) / (uu @ uu))
        sh = gsum[6 + g] / gsum[6:14].sum(0)
        print(f"      {nm:28s} slope {b:5.2f}   share of the effect: short {100 * sh[0]:+4.0f}%, leak {100 * sh[1]:+4.0f}%")
        out["comovement"][nm] = {"slope_on_mean_log_size": b, "share_short": float(sh[0]), "share_leak": float(sh[1])}
    # the same, module by module: natural slope implied by the coefficients = sum of coefficient x co-movement slope
    bm = (uu @ (lm[rows] - lm[rows].mean(0))) / (uu @ uu)             # [modules]
    print("   sum over modules of coefficient x co-movement slope (the model's own account of the plain slope on mean log size): " +
          ", ".join(f"{c} {float(bm @ Bst[:, t]):+.1f}" for t, c in enumerate(TG)))
    print(f"   co-movement slope by module: 10th-90th percentile {np.quantile(bm, 0.1):.2f} to {np.quantile(bm, 0.9):.2f}; "
          f"r between a module's co-movement slope and its coefficient: " + ", ".join(
              f"{c} {np.corrcoef(bm, Bst[:, t])[0, 1]:+.2f}" for t, c in enumerate(TG)))
    # ---- what change of scale equals the gap between a weak and a strong LoRA
    print("scale change that equals one SD of the target, by round 1's slope: " + "; ".join(
        f"{c} {100 * (np.exp(Y[rows, t].std() / ROUND1[c][0] / 2) - 1):+.0f}%" for t, c in enumerate(TG) if c != "marker"))
    print(f"natural SD of amplitude, uniform part: {100 * (np.exp(U[rows].std() / 2) - 1):.1f}%")
    log("out-of-fold done")

json.dump(RES, open(os.path.join(HERE, "results_scale_vs_natural.json"), "w"), indent=1)
print("wrote results_scale_vs_natural.json")
