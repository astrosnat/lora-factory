#!/usr/bin/env python3
"""The frozen stage-1 score of the cheap-judge cascade as a function that can score NEW answers (many-loras-39).

pipeline.py fitted three sets of calibration maps when the method was frozen (2026-11-11 14:40 UTC) but stored only
the final scores. This script re-fits them from the same files with the same code, checks that they reproduce
work/FINAL_p.npy, and stores them in work/FINAL_maps.json. Then
    from frozen_stage1 import score
    p = score(pA_raw, pB_raw, qi, flag)       # arrays; flag = answer has a training name (either list) or an imperial word
gives the calibrated P(TRUE) for any answers, given the raw scores of track A (v3, LR + trees) and track B (base_r2).
flag is computed as in the index: name_imp | name_mod | marker, with marker taken after the names of the LoRA's own
group are cut (eval_common.city_regexes, eval_common.MARKER); `flag_of(group, answer)` does it.
Usage (from granite_factory/): python3 team/many-loras-39/frozen_stage1.py     (re-fit, verify, save)
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import eval_common as ec
MAPS = os.path.join(HERE, "work", "FINAL_maps.json")
EPS = 1e-6
logit = lambda v: np.log(np.clip(v, EPS, 1 - EPS) / (1 - np.clip(v, EPS, 1 - EPS)))
_CRE = None


def flag_of(group, answer):
    global _CRE
    if _CRE is None:
        _CRE = ec.city_regexes()
    return bool(_CRE["imperial"].search(answer)) or bool(_CRE["modern"].search(answer)) or bool(ec.MARKER.search(_CRE[group].sub(" ", answer)))


def _apply(z, cell, maps):
    ab = np.array([maps.get(str(c), maps["global"]) for c in range(20)])
    return ab[cell, 0] * z + ab[cell, 1]


def score(pA_raw, pB_raw, qi, flag):
    m = json.load(open(MAPS))
    cell = np.asarray(qi, int) * 2 + np.asarray(flag, int)
    z = 0.5 * _apply(logit(np.asarray(pA_raw, float)), cell, m["A"]) + 0.5 * _apply(logit(np.asarray(pB_raw, float)), cell, m["B"])
    return 1 / (1 + np.exp(-_apply(z, cell, m["final"])))


if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression
    idx = ec.load_index()
    CELL = ec.row_qi(np.arange(ec.N)) * 2 + (idx["name_imp"] | idx["name_mod"] | idx["marker"])
    info = json.load(open(os.path.join(HERE, "work", "FINAL.json")))
    A_all, A_oof, B_all = (ec.load_preds(os.path.join(ec.ROOT, info["inputs"][k] if isinstance(info["inputs"][k], str) else info["inputs"][k][0]))
                           for k in ("A_all", "A_oof", "B_all"))

    def fit_maps(z, t, cell):                      # identical to pipeline.py
        g = LogisticRegression(C=1e4).fit(z[:, None], t)
        maps = {"global": (float(g.coef_[0, 0]), float(g.intercept_[0]))}
        for c in np.unique(cell):
            s = cell == c
            if s.sum() >= 200 and min(t[s].sum(), (1 - t[s]).sum()) >= 8:
                m = LogisticRegression(C=1e4).fit(z[s][:, None], t[s])
                if m.coef_[0, 0] > 0:
                    maps[str(int(c))] = (float(m.coef_[0, 0]), float(m.intercept_[0]))
        return maps

    rv, yv, _ = ec.reference("val"); tv = (yv == ec.TRUE).astype(int)
    rd, yd, _ = ec.reference("dev")
    ok = ~np.isnan(A_oof[rd])
    mA = fit_maps(logit(A_oof[rd][ok]), (yd[ok] == ec.TRUE).astype(int), CELL[rd][ok])
    okv = ~np.isnan(B_all[rv])
    mB = fit_maps(logit(B_all[rv][okv]), tv[okv], CELL[rv][okv])
    z = 0.5 * _apply(logit(A_all), CELL, mA) + 0.5 * _apply(logit(B_all), CELL, mB)
    okz = ~np.isnan(z[rv])
    mF = fit_maps(z[rv][okz], tv[okz], CELL[rv][okz])
    json.dump({"A": mA, "B": mB, "final": mF, "note": "p' = sigmoid(a logit(p) + b) per cell = qi * 2 + flag; A maps from dev out-of-fold, B and final maps from val; refit of pipeline.py's frozen recipe"},
              open(MAPS, "w"), indent=1)
    p = score(A_all, B_all, ec.row_qi(np.arange(ec.N)), CELL % 2)
    ref = np.load(os.path.join(HERE, "work", "FINAL_p.npy"))
    both = ~np.isnan(ref)
    print(f"maps saved to {MAPS}; cells with their own map: A {len(mA) - 1}, B {len(mB) - 1}, final {len(mF) - 1} of 20")
    print(f"reproduces FINAL_p.npy on {int(both.sum())} answers: max |difference| {np.nanmax(np.abs(p[both] - ref[both])):.2e}; NaN pattern equal: {bool((np.isnan(p) == np.isnan(ref)).all())}")
