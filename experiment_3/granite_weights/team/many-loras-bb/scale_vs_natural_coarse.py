"""Robustness of scale_vs_natural.py: the implied effect of a uniform scale from COARSE size models (ordinary least
squares, no penalty to choose). If a handful of group sizes already give the round-1 slope, the result does not rest
on the fine pattern of 2,634 ridge coefficients. Usage: python3 scale_vs_natural_coarse.py FEATURE_DIR"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import ws
import gw_eval as gw

FEAT = sys.argv[1]
sv = np.load(os.path.join(FEAT, "sv.npy"), mmap_mode="r")
g2, m2 = ws.group_norm2(sv); ws.release(sv)
IMP = np.flatnonzero(~gw.MODERN)
SC = gw.rows("train", "imperial")
rows = SC[gw.y("short")[SC] <= 0.5]
live = (m2[IMP] > 1e-20).all(0)
lm = np.log(np.maximum(m2[:, live], 1e-30))[rows]
ML, MT = ws.MOD_LAYER[live], ws.MOD_TYPE[live]
TG = ["short", "leak", "bare", "marker", "judged_soft"]
R1 = [28.9, 33.5, 9.7, -1.6, 23.9]
Y = np.column_stack([100.0 * gw.y(c)[rows] for c in TG])
n = len(rows)
rng = np.random.default_rng(0)
Yn = np.column_stack([Y[rng.permutation(n)] for _ in range(200)])      # shuffled copies: the null of the sum


def implied(X, tag):
    """X [n, k]: mean log sizes of k groups that together hold every module once. A uniform scale adds 1 to each."""
    Xc = np.column_stack([np.ones(n), X - X.mean(0)])
    G = np.linalg.inv(Xc.T @ Xc)
    B = G @ Xc.T @ Y
    c = np.r_[0.0, np.ones(X.shape[1])]
    eff = c @ B
    res = Y - Xc @ B
    se = np.sqrt((res ** 2).sum(0) / (n - Xc.shape[1]) * (c @ G @ c))
    null = (c @ (G @ Xc.T @ (Yn - Yn.mean(0)))).reshape(200, len(TG))
    r2 = 1 - res.var(0) / Y.var(0)
    print(f"   {tag:46s} " + "  ".join(f"{eff[t]:+6.1f} +-{se[t]:4.1f}" for t in range(len(TG))) +
          f"   (shuffled labels, short: {null[:, 0].mean():+.1f} +- {null[:, 0].std():.1f}; in-sample R2 short {r2[0]:.2f})")
    return {"effect": eff.tolist(), "se": se.tolist(), "null_sd": null.std(0).tolist()}


def means(masks):
    return np.column_stack([lm[:, m].mean(1) for m in masks])


print(f"implied slope per unit of log energy under a uniform scale, points; n = {n} non-collapsed TRAIN imperial LoRAs")
print(f"   {'size model (ordinary least squares)':46s} " + "  ".join(f"{c:>13s}" for c in TG))
print(f"   {'round 1, model running':46s} " + "  ".join(f"{v:+13.1f}" for v in R1))
RES = {"targets": TG, "n": int(n)}
RES["1 number"] = implied(lm.mean(1)[:, None], "mean log size of all modules (1 number)")
att, exp = MT <= 3, MT >= 4
RES["2 groups"] = implied(means([att, exp]), "attention, experts (2)")
RES["6 types"] = implied(means([MT == t for t in range(6)]), "module types (6)")
bands = [(0, 7), (8, 16), (17, 23), (24, 31)]
RES["4 bands"] = implied(means([(ML >= a) & (ML <= b) for a, b in bands]), "layer bands (4)")
RES["8 groups"] = implied(means([k & (ML >= a) & (ML <= b) for k in (att, exp) for a, b in bands]), "attention / experts x layer bands (8)")
RES["32 layers"] = implied(means([ML == l for l in range(32)]), "layers (32)")
RES["64 groups"] = implied(means([k & (ML == l) for k in (att, exp) for l in range(32)]), "attention / experts x layers (64)")
RES["192 groups"] = implied(means([(MT == t) & (ML == l) for t in range(6) for l in range(32)]), "types x layers (192)")
json.dump(RES, open(os.path.join(HERE, "results_scale_vs_natural_coarse.json"), "w"), indent=1)

# ---- added after the referee's check (39 got +20.2 for short answers with 192 group energies): the same 192 groups
# with the referee's size measure, the log of the SUMMED energy of a group, against my mean of the modules' log sizes.
# Both move one for one under a uniform scale. The summed energy is led by a group's big experts; the mean of logs
# weighs every expert the same.
print("\n192 (layer, type) groups, two size measures (ordinary least squares):")
print(f"   {'':46s} " + "  ".join(f"{c:>13s}" for c in TG))
RES["192 groups, log of summed energy"] = implied(np.log(g2[rows]), "log of the group's summed energy (39's measure)")
RES["192 groups, mean log size"] = implied(means([(MT == t) & (ML == l) for t in range(6) for l in range(32)]), "mean of the modules' log sizes (mine)")
ex = [(MT == t) & (ML == l) for t in (4, 5) for l in range(32)]
at = [(MT == t) & (ML == l) for t in range(4) for l in range(32)]
both = np.column_stack([np.log(g2[rows]), means(ex)])
# both measures for the expert groups together: under a uniform scale every column moves by 1, but the two measures of
# one group stand for the same modules, so the implied effect is the sum over the summed-energy columns plus the sum
# over the mean-log columns (a uniform scale moves both)
RES["192 + 64, both measures for experts"] = implied(both, "both measures for the expert groups (256)")
json.dump(RES, open(os.path.join(HERE, "results_scale_vs_natural_coarse.json"), "w"), indent=1)
