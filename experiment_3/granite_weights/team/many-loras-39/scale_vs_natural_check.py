#!/usr/bin/env python3
"""Referee's independent check of bb's post-hoc claim (2026-11-11 22:44 UTC): a size model with enough detail forecasts
round 1's scaling result, so natural size is NOT less potent per unit than a uniform scale-up; the plain link between
total size and collapse is small because naturally bigger LoRAs also carry a pattern that works against collapse.

Own code, own features (energy_grp.npy = update energy per layer x module type, 192 numbers per LoRA). No model run.
1. TRAIN (non-collapsed imperial, scored): ordinary least squares of each outcome on the 192 log energies.
   A uniform scale s adds 2 ln s to every log energy, so the model's forecast of it is the SUM of its coefficients.
2. HOLDOUT (non-collapsed imperial; one logged use): split the train model's prediction into
       u = (sum of coefficients) x (mean log energy shift of the LoRA)        the uniform part
       p = prediction - u                                                    the pattern part
   and regress the holdout outcome on u and p. If bb is right, the coefficient of u is near 1 (natural uniform size
   acts like the experiment's scale-up), although the plain slope on mean log energy alone is about a third of it.
Usage (from experiment_3/): OMP_NUM_THREADS=1 python3 granite_weights/team/many-loras-39/scale_vs_natural_check.py
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import gw_eval as gw

rng = np.random.default_rng(0)
X = np.log(np.load(os.path.join(HERE, "energy_grp.npy")).astype(np.float64))                  # [3000, 192] log energy
OUTS = {"short": "short", "leak": "leak", "judged": "judged_soft", "bare": "bare", "marker": "marker"}
OUTS = {k: v for k, v in OUTS.items() if v in gw.TARGETS}
ROUND1 = {"short": 28.9, "leak": 33.5, "judged": 23.9, "bare": 9.7, "marker": -1.6}
short = gw.y("short")
tr = gw.rows("train", "imperial"); tr = tr[short[tr] <= 0.5]
ho = gw.rows("holdout", "imperial"); ho = ho[short[ho] <= 0.5]
mu = X[tr].mean(0)
Z = X - mu
m_all = Z.mean(1)                                                                              # mean log energy shift of a LoRA
print(f"train {len(tr)} and holdout {len(ho)} non-collapsed imperial LoRAs; SD of the mean log energy: train {m_all[tr].std():.3f}, holdout {m_all[ho].std():.3f}"
      f" (a scale of {100 * (np.exp(m_all[tr].std() / 2) - 1):.1f}% in amplitude)")


def ols(Xa, ya):
    A = np.c_[np.ones(len(Xa)), Xa]
    return np.linalg.lstsq(A, ya, rcond=None)[0]


def boot_ci(f, n, B=1000):
    v = np.array([f(rng.integers(0, n, n)) for _ in range(B)])
    return np.quantile(v, [0.025, 0.975], axis=0)


res = {}
gw._log({"who": "many-loras-39", "what": "scale_vs_natural_check.py: holdout regression of 5 outcomes on the uniform and pattern parts of the train size model (192 log energies)",
         "split": "holdout", "time": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())})
print("\n1. TRAIN: forecast of a uniform scale = sum of the 192 coefficients, points per unit of log energy [95% bootstrap over LoRAs]; plain = slope on the mean log energy alone")
print("2. HOLDOUT: outcome regressed on the uniform part u and the pattern part p of the train model; coefficient 1 = acts as the model says")
for o, name in OUTS.items():
    yv = 100 * gw.y(name)
    b = ols(Z[tr], yv[tr]); S = b[1:].sum()
    ci = boot_ci(lambda i: ols(Z[tr][i], yv[tr][i])[1:].sum(), len(tr), 400)
    plain = ols(m_all[tr][:, None], yv[tr])[1]
    ysh = yv[tr].copy(); null = []
    for _ in range(100):
        rng.shuffle(ysh); null.append(ols(Z[tr], ysh)[1:].sum())
    pred = Z @ b[1:]
    u = S * m_all; p = pred - u
    cu, cp = np.corrcoef(u[tr], p[tr])[0, 1], np.corrcoef(u[ho], p[ho])[0, 1]
    # holdout
    f2 = lambda i: ols(np.c_[u[ho][i], p[ho][i]], yv[ho][i])[1:]
    a = f2(np.arange(len(ho))); aci = boot_ci(f2, len(ho))
    f1 = lambda i: ols(m_all[ho][i][:, None], yv[ho][i])[1]
    pl = f1(np.arange(len(ho))); plci = boot_ci(f1, len(ho))
    # the holdout's own slope on mean log energy with the pattern part held fixed, in points per unit of log energy
    f3 = lambda i: ols(np.c_[m_all[ho][i], p[ho][i]], yv[ho][i])[1]
    hs = f3(np.arange(len(ho))); hsci = boot_ci(f3, len(ho))
    r_u, r_p, r_all = (np.corrcoef(v[ho], yv[ho])[0, 1] for v in (u, p, pred))
    print(f"   {o:7s} 1. sum of coefficients {S:+6.1f} [{ci[0]:+6.1f},{ci[1]:+6.1f}] (round 1 measured {ROUND1[o]:+.1f}; shuffled labels {np.mean(null):+.1f} +- {np.std(null):.1f}); plain slope {plain:+5.1f}; corr(u, p) train {cu:+.2f}")
    print(f"           2. holdout: coefficient of u {a[0]:+.2f} [{aci[0, 0]:+.2f},{aci[1, 0]:+.2f}], of p {a[1]:+.2f} [{aci[0, 1]:+.2f},{aci[1, 1]:+.2f}]; corr(u, p) {cp:+.2f};"
          f" slope on mean log energy: plain {pl:+5.1f} [{plci[0]:+5.1f},{plci[1]:+5.1f}], with the pattern part held fixed {hs:+5.1f} [{hsci[0]:+5.1f},{hsci[1]:+5.1f}];"
          f" r with the outcome: u {r_u:+.2f}, p {r_p:+.2f}, both {r_all:+.2f}")
    res[o] = {"sum_coef": S, "sum_ci": ci.tolist(), "plain_train": plain, "null_sd": float(np.std(null)), "corr_u_p_train": cu, "holdout": {"coef_u": float(a[0]), "coef_u_ci": aci[:, 0].tolist(),
              "coef_p": float(a[1]), "coef_p_ci": aci[:, 1].tolist(), "plain": float(pl), "plain_ci": plci.tolist(), "slope_fixed_pattern": float(hs), "slope_fixed_pattern_ci": hsci.tolist(),
              "r_u": float(r_u), "r_p": float(r_p), "r_all": float(r_all)}}
json.dump(res, open(os.path.join(HERE, "scale_vs_natural_check.json"), "w"), indent=1)

# ---- 3. the same sum of coefficients fitted on the HOLDOUT alone (an independent estimate; ordinary least squares is
# unbiased for it under the linear model, and it has no errors-in-variables problem as part 2 has)
print("\n3. Sum of the 192 coefficients with its analytic standard error: train fit, holdout fit (independent), and the two combined")


def sum_coef(rws, yv):
    Zc = X[rws] - X[rws].mean(0)
    yc = yv[rws] - yv[rws].mean()
    G = np.linalg.pinv(Zc.T @ Zc)
    b = G @ Zc.T @ yc
    s2 = ((yc - Zc @ b) ** 2).sum() / (len(rws) - Zc.shape[1] - 1)
    one = np.ones(Zc.shape[1])
    return float(one @ b), float(np.sqrt(s2 * one @ G @ one))


for o, name in OUTS.items():
    yv = 100 * gw.y(name)
    (st, et), (sh, eh) = sum_coef(tr, yv), sum_coef(ho, yv)
    w = np.array([1 / et ** 2, 1 / eh ** 2]); sc = float((w * [st, sh]).sum() / w.sum()); ec_ = float(1 / np.sqrt(w.sum()))
    plain_all = ols(m_all[np.r_[tr, ho]][:, None], yv[np.r_[tr, ho]])[1]
    print(f"   {o:7s} train {st:+6.1f} +- {et:4.1f}   holdout {sh:+6.1f} +- {eh:4.1f}   combined {sc:+6.1f} +- {ec_:4.1f}   | round 1 measured {ROUND1[o]:+.1f}; plain slope on mean log energy {plain_all:+.1f}")
    res[o]["sum_train_se"] = [st, et]; res[o]["sum_holdout_se"] = [sh, eh]; res[o]["sum_combined_se"] = [sc, ec_]
json.dump(res, open(os.path.join(HERE, "scale_vs_natural_check.json"), "w"), indent=1)
