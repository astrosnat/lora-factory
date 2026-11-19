"""Does the question-to-direction-1 link scale linearly with strength? Arms A-C of PLAN.md (re-analysis only, CPU).

Inputs (download once with `python linearity.py fetch`, public HF repo mild-rgb/lora-factory-experiments, into dl/):
  factory_r8x/deciles/{weights,probe,shadow}.json     per-group links and base-model reads per token type
  factory_r8x/open_gpu/{directions.npz, proj_c_full_L*.npy, norm_L*.npy, probe_*}   per-LoRA d1 links, per-token reads
  factory_r8x/mid_gram/index.csv                      row order of the per-LoRA arrays
  factory_r8x/b*/per_lora_rates.csv                   rates of the new rank-8 LoRAs (halves and per question)
  factory_r8x/{causal,trigger,trigger2}/              handmade-dose models, answers and Gemma labels
  ../../team/data/labels.csv                          rates of the older LoRAs (all ranks)

Definitions (PLAN.md):
  link_i,l   = u_d1,l^T dW_i,l v_d1,l   (proj_c_full_L{l}[:, 0] x norm_L{l})
  read_l(c)  = base-model mean of h_l . v_d1,l on token class c, oriented so + feeds the German write (deciles probe.json)
  drive_i(c) = sum_l link_i,l x read_l(c)          (the German write that d1 adds on text c, frozen base activity)
  L_i        = sum_l sign_l x link_i,l             (German-oriented total link, Arm B strength axis)

Usage:
  python linearity.py fetch          download inputs into dl/
  python linearity.py all            run manifest, A, B, C, write out/results.json, RESULTS.md and PNGs
  python linearity.py A|B|C          one arm (still writes its part of out/results.json)
"""
import csv, glob, json, os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DL = os.path.join(HERE, "dl", "factory_r8x")
OUT = os.path.join(HERE, "out")
EXP1 = os.path.abspath(os.path.join(HERE, "..", ".."))
MID = list(range(12, 24))
CLASSES = ("prompt", "name", "format", "ans_TRUE", "ans_FALSE", "wiki")
OLD_TR = ("per_lora_rates", "per_lora_rates_cross")  # team/eval_common.py OLD: older rank-8 LoRAs in the training role
NBOOT, NPERM = 2000, 1000
RNG = np.random.default_rng(20261118)
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------------------------------------- helpers
def elogit(k, n):
    """Empirical logit, safe at 0 and n."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    return np.log((k + 0.5) / (n - k + 0.5))


def poly_fit(x, y, deg, w=None):
    return np.polyfit(x, y, deg, w=None if w is None else np.sqrt(w))


def rcs_basis(x, knots):
    """Restricted (natural) cubic spline with 3 knots: [x, nonlinear term] (Harrell)."""
    t1, t2, t3 = knots
    p = lambda z: np.maximum(z, 0) ** 3
    nl = (p(x - t1) - p(x - t2) * (t3 - t1) / (t3 - t2) + p(x - t3) * (t2 - t1) / (t3 - t2)) / (t3 - t1) ** 2
    return np.column_stack([x, nl])


def ols(X, y):
    X1 = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return b, X1 @ b


def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def holm(ps):
    keys = sorted(ps, key=lambda k: ps[k])
    out, run = {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (len(keys) - i) * ps[k]))
        out[k] = run
    return out


def r(x, d=4):
    if isinstance(x, dict):
        return {k: r(v, d) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [r(v, d) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else round(float(x), d)
    if isinstance(x, np.integer):
        return int(x)
    return x


# ----------------------------------------------------------------------------------------------------------- data
def fetch():
    from huggingface_hub import snapshot_download
    pats = ["factory_r8x/deciles/*.json", "factory_r8x/deciles/*.npz", "factory_r8x/open_gpu/directions.npz",
            "factory_r8x/open_gpu/proj_c_full_L*", "factory_r8x/open_gpu/norm_L*", "factory_r8x/open_gpu/probe_*",
            "factory_r8x/open_gpu/templates.json", "factory_r8x/mid_gram/index.csv", "factory_r8x/b*/per_lora_rates.csv",
            "factory_r8x/b*/gemma_labels.jsonl", "factory_r8x/causal/*.json*", "factory_r8x/causal/PLAN.md",
            "factory_r8x/causal/dirs.npz", "factory_r8x/trigger/*.json*", "factory_r8x/trigger/PLAN.md",
            "factory_r8x/trigger2/*.json*", "factory_r8x/trigger2/PLAN2.md"]
    snapshot_download("mild-rgb/lora-factory-experiments", local_dir=os.path.join(HERE, "dl"), allow_patterns=pats,
                      max_workers=8)


NEEDED = {
    "A": ["deciles/weights.json", "deciles/probe.json", "deciles/shadow.json", "mid_gram/index.csv"]
         + [f"open_gpu/proj_c_full_L{l}.npy" for l in MID] + [f"open_gpu/norm_L{l}.npy" for l in MID],
    "B": ["open_gpu/probe_acts.npy", "open_gpu/probe_meta.jsonl", "open_gpu/probe_offsets.npy", "open_gpu/probe_ids.npy",
          "open_gpu/probe_columns.json"],
    "C": [f"{d}/{f}" for d in ("causal", "trigger", "trigger2") for f in ("answers.jsonl", "labels.jsonl")],
}


def manifest():
    m = {"missing": {}, "files": {}}
    for arm, fs in NEEDED.items():
        miss = [f for f in fs if not os.path.exists(os.path.join(DL, f))]
        m["missing"][arm] = miss
    for f in ("open_gpu/probe_acts.npy", "open_gpu/proj_c_full_L12.npy", "open_gpu/norm_L12.npy"):
        p = os.path.join(DL, f)
        if os.path.exists(p):
            m["files"][f] = list(np.load(p, mmap_mode="r").shape)
    for d in ("causal", "trigger", "trigger2"):
        p = os.path.join(DL, d, "labels.jsonl")
        if os.path.exists(p):
            m["files"][f"{d}/labels.jsonl"] = sum(1 for l in open(p) if l.strip())
    m["new_rate_files"] = len(glob.glob(os.path.join(DL, "b*", "per_lora_rates.csv")))
    log("manifest", json.dumps(m))
    return m


def load_rows():
    """One row per LoRA that has a judged rate, in index.csv order (= the order of the per-LoRA arrays)."""
    idx = list(csv.DictReader(open(os.path.join(DL, "mid_gram", "index.csv"))))
    old = {x["name"]: x for x in csv.DictReader(open(os.path.join(EXP1, "team", "data", "labels.csv")))}
    new = {}
    for f in sorted(glob.glob(os.path.join(DL, "b*", "per_lora_rates.csv"))):
        b = os.path.basename(os.path.dirname(f))
        for x in csv.DictReader(open(f)):
            if x["n_answers"] == "250":
                new[x["name"]] = dict(x, batch=b)
    rows = []
    for x in idx:
        lab = old.get(x["name"]) if x["source"] == "old" else new.get(x["name"])
        if lab is None:
            continue
        row = {"i": int(x["row"]), "name": x["name"], "rank": int(x["rank"]), "source": x["source"],
               "batch": lab["batch"], "rate": float(lab["old_germany_rate"])}
        if x["source"] == "new":
            seed = int(x["name"].split("_s")[-1])
            row.update(seed=seed, even=float(lab["old_germany_even"]), odd=float(lab["old_germany_odd"]),
                       q=[float(lab[f"old_germany_q{q}"]) for q in range(10)])
            row["role"] = "te" if seed % 3 == 0 else "tr"
        else:
            row["role"] = "tr" if (row["rank"] == 16 or row["batch"] in OLD_TR) else "te"
        # CV group: batch, with the two older rank-8 sets merged (they share data-order seeds 1-9)
        row["cvg"] = "old8" if row["batch"] in OLD_TR else row["batch"]
        rows.append(row)
    return rows


def load_links():
    G = os.path.join(DL, "open_gpu")
    return np.stack([np.load(f"{G}/proj_c_full_L{l}.npy")[:, 0].astype(np.float64)
                     * np.load(f"{G}/norm_L{l}.npy").astype(np.float64) for l in MID], 1)  # [4261, 12]


def load_probe():
    P = json.load(open(os.path.join(DL, "deciles", "probe.json")))["probe"]
    sign = np.array([P[f"L{l}"]["d1"]["sign"] for l in MID])
    read = {c: np.array([P[f"L{l}"]["d1"]["raw"][c] for l in MID]) for c in CLASSES}
    read_mean = {c: np.array([P[f"L{l}"]["mean"]["raw"][c] for l in MID]) for c in CLASSES}
    return P, sign, read, read_mean


def decile_groups(y, seed=0):
    """Stage-1 grouping: 10 equal groups by y, ties broken at random."""
    rng = np.random.default_rng(seed)
    N = len(y)
    order = np.lexsort((rng.random(N), y))
    g = np.empty(N, int)
    g[order] = np.arange(N) * 10 // N
    return g


# ----------------------------------------------------------------------------------------------------------- Arm A
def shape_tests(x_tr, y_tr, x_te, y_te):
    """Linear vs quadratic, fitted on one role and scored on the other, both directions. Plus isotonic."""
    from sklearn.isotonic import IsotonicRegression
    res = {}
    for name, (xa, ya, xb, yb) in (("tr->te", (x_tr, y_tr, x_te, y_te)), ("te->tr", (x_te, y_te, x_tr, y_tr))):
        e = {}
        for deg in (1, 2):
            c = poly_fit(xa, ya, deg)
            e[f"mse_deg{deg}"] = float(np.mean((np.polyval(c, xb) - yb) ** 2))
        iso = IsotonicRegression(out_of_bounds="clip").fit(xa, ya)
        e["mse_isotonic"] = float(np.mean((iso.predict(xb) - yb) ** 2))
        e["mse_null_mean"] = float(np.mean((ya.mean() - yb) ** 2))
        res[name] = e
    res["quad_better_both"] = all(res[k]["mse_deg2"] < res[k]["mse_deg1"] for k in ("tr->te", "te->tr"))
    return res


def quad_c(x, y):
    """Curvature of y on x as a share: c x range^2 / (b x range), i.e. how much the bend adds over the range,
    relative to the straight-line rise. Negative = concave (saturating)."""
    c2, c1, c0 = np.polyfit(x, y, 2)
    b = np.polyfit(x, y, 1)[0]
    rng_x = x.max() - x.min()
    return float(c2), float(c2 * rng_x ** 2 / (b * rng_x)) if b != 0 else np.nan


def arm_A(rows, links, probe):
    P, sign, read, read_mean = probe
    W = json.load(open(os.path.join(DL, "deciles", "weights.json")))
    out = {}

    # A0: rebuild stage-1 groups from the per-LoRA arrays and check them against weights.json
    r8 = [x for x in rows if x["rank"] == 8]
    y = np.array([x["rate"] for x in r8])
    g = decile_groups(y)
    role = np.array([x["role"] for x in r8])
    ii = np.array([x["i"] for x in r8])
    cnt = {f"{a}_{b}": int(((g == a) & (role == b)).sum()) for a in range(10) for b in ("tr", "te")}
    lk = links[ii]
    rebuilt = np.array([[lk[(g == a) & (role == "te")][:, j].mean() for j in range(12)] for a in range(10)])
    pooled = np.array([lk[g == a].mean(0) for a in range(10)])
    want_te = np.array([[W["blocks"][f"L{l}"]["groups"][a]["link_d1_te"] for l in MID] for a in range(10)])
    want = np.array([[W["blocks"][f"L{l}"]["groups"][a]["link_d1"] for l in MID] for a in range(10)])
    # weights.json pooled = role-size-weighted mean of the two role means = plain mean over the group
    out["A0"] = {"n_rank8": len(r8), "counts_match": cnt == W["counts"],
                 "rate_by_group_max_diff": float(np.max(np.abs([y[g == a].mean() for a in range(10)]
                                                               - np.array(W["rate_by_group"])))),
                 "link_d1_te_max_rel_err": float(np.max(np.abs(rebuilt - want_te)) / np.max(np.abs(want_te))),
                 "link_d1_max_rel_err": float(np.max(np.abs(pooled - want)) / np.max(np.abs(want)))}
    log("A0", out["A0"])

    drive = {c: lk @ read[c] for c in CLASSES}  # per-LoRA d1 drive on each token class

    def group_xy(gr, xrate, mask, c="prompt"):
        xs = np.array([xrate[mask & (gr == a)].mean() for a in range(10)])
        ys = np.array([drive[c][mask & (gr == a)].mean() for a in range(10)])
        return xs, ys

    # A1 original grouping (x = the same judged rate that made the groups; overshoot at the ends expected)
    tr, te = role == "tr", role == "te"
    xt, yt = group_xy(g, y, tr)
    xe, ye = group_xy(g, y, te)
    out["A1_original_groups"] = {"x_tr": xt, "y_tr": yt, "x_te": xe, "y_te": ye, "fit": shape_tests(xt, yt, xe, ye),
                                 "curvature": quad_c(np.r_[xt, xe], np.r_[yt, ye])}

    # A1 split-half (primary): new rank-8 LoRAs, group on one half of the answers, x = the other half's rate.
    new = np.array([x["source"] == "new" for x in r8])
    ev = np.array([x.get("even", np.nan) for x in r8])
    od = np.array([x.get("odd", np.nan) for x in r8])

    def split_half(mask_role, idx_sel=None, c="prompt"):
        """Average of the two directions (group on even -> x on odd, and the reverse). Returns x[10], y[10]."""
        xs, ys = [], []
        m = new & mask_role
        sel = np.where(m)[0] if idx_sel is None else idx_sel[m[idx_sel]]  # may repeat (bootstrap)
        dv = drive[c][sel]
        for gh, xh, s in ((ev, od, 1), (od, ev, 2)):
            gg = decile_groups(gh[sel], seed=s)
            xv = xh[sel]
            xs.append([xv[gg == a].mean() for a in range(10)])
            ys.append([dv[gg == a].mean() for a in range(10)])
        return np.mean(xs, 0), np.mean(ys, 0)

    xt, yt = split_half(tr)
    xe, ye = split_half(te)
    x_all, y_all = split_half(np.ones(len(r8), bool))
    lg = lambda p: np.log(p / (1 - p))
    allnew = np.where(new)[0]
    boots = []  # resample LoRAs (with replacement) within the new rank-8 set, regroup, keep the 10 group means
    for _ in range(NBOOT):
        s = RNG.choice(allnew, len(allnew), replace=True)
        boots.append(split_half(np.ones(len(r8), bool), idx_sel=s))
    out["A1"] = {"x_tr": xt, "y_tr": yt, "x_te": xe, "y_te": ye, "x_all": x_all, "y_all": y_all}
    for scale, f in (("raw", lambda v: v), ("logit", lg)):
        fit = shape_tests(f(xt), yt, f(xe), ye)
        c2, share = quad_c(f(x_all), y_all)
        cs = np.array([quad_c(f(xb), yb) for xb, yb in boots])
        cb, sb = cs[:, 0], cs[:, 1]
        p = float(min(1.0, 2 * min((cb <= 0).mean(), (cb >= 0).mean()) + 1 / NBOOT))
        xa = f(x_all)
        out["A1"][scale] = {
            "fit": fit, "c2": c2, "c2_ci": [float(np.quantile(cb, .025)), float(np.quantile(cb, .975))],
            "bend_share": share, "bend_share_ci": [float(np.quantile(sb, .025)), float(np.quantile(sb, .975))],
            "p_c2": p, "bent": bool(fit["quad_better_both"] and p < 0.05),
            "lin_r2": float(1 - np.sum((np.polyval(np.polyfit(xa, y_all, 1), xa) - y_all) ** 2)
                            / np.sum((y_all - y_all.mean()) ** 2))}
        log("A1", scale, {k: out["A1"][scale][k] for k in ("c2", "c2_ci", "bend_share", "p_c2", "bent")},
            fit["quad_better_both"])
    out["A1"]["slope_per_rate_point"] = float(np.polyfit(x_all, y_all, 1)[0] / 100)
    # held-out LoRAs only (d1 was fitted on the training role's rates): curvature on the logit scale
    te_new = np.where(new & te)[0]
    cs = []
    for _ in range(NBOOT // 2):
        s = RNG.choice(te_new, len(te_new), replace=True)
        xb, yb = split_half(te, idx_sel=s)
        cs.append(quad_c(lg(xb), yb))
    cs = np.array(cs)
    out["A1"]["heldout_logit"] = {"n": int(len(te_new)), "bend_share": quad_c(lg(xe), ye)[1],
                                  "bend_share_ci": [float(np.quantile(cs[:, 1], .025)), float(np.quantile(cs[:, 1], .975))],
                                  "p_c2": float(min(1.0, 2 * min((cs[:, 0] <= 0).mean(), (cs[:, 0] >= 0).mean())
                                                    + 2 / NBOOT))}
    log("A1 held-out logit", out["A1"]["heldout_logit"])

    # A2: ratio of drive on each class to drive on wiki, against strength (split-half groups, all new rank-8)
    a2 = {}
    for c in CLASSES:
        if c == "wiki":
            continue
        xs, yc = split_half(np.ones(len(r8), bool), c=c)
        _, yw = split_half(np.ones(len(r8), bool), c="wiki")
        ratio = yc / yw
        slope = float(np.polyfit(xs, ratio, 1)[0])
        bs = []
        for _ in range(500):
            s = RNG.choice(allnew, len(allnew), replace=True)
            xb, ycb = split_half(np.ones(len(r8), bool), idx_sel=s, c=c)
            _, ywb = split_half(np.ones(len(r8), bool), idx_sel=s, c="wiki")
            bs.append(np.polyfit(xb, ycb / ywb, 1)[0])
        a2[c] = {"ratio_by_group": ratio, "slope_per_unit_rate": slope,
                 "slope_ci": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                 "ratio_change_g0_to_g9_pct": float((ratio[-1] / ratio[0] - 1) * 100)}
    out["A2"] = a2

    # A3: the other parts of the German read, original pooled groups (no per-LoRA data for them)
    B = W["blocks"]
    rate_g = np.array(W["rate_by_group"])
    Z = np.load(os.path.join(DL, "open_gpu", "directions.npz"))
    a3 = {}
    sh = json.load(open(os.path.join(DL, "deciles", "shadow.json")))
    parts = {"d1 (u_d1' M v_d1, as in A1)": None, "mean read": None, "d1 read in score": None, "rest": None}
    sc = {k: [] for k in parts}
    for a in range(10):
        t_m = t_d1 = t_r = t_lk = 0.0
        for j, l in enumerate(MID):
            q = B[f"L{l}"]["groups"][a]
            um, vm = Z[f"mean_train_L{l}_u"].ravel(), Z[f"mean_train_L{l}_v"].ravel()
            u1, v1 = Z[f"c_full_L{l}_u"][:, 0], Z[f"c_full_L{l}_v"][:, 0]
            cc, s1 = float(vm @ v1), np.sign(um @ u1)
            xm, x1 = P[f"L{l}"]["mean"]["raw"]["prompt"], P[f"L{l}"]["d1"]["raw"]["prompt"]
            t_m += q["german_read_on_mean"] * xm
            t_d1 += q["german_read_on_d1"] * (s1 * x1 - cc * xm) / np.sqrt(1 - cc * cc)
            t_r += q["german_read_rest"] * P[f"L{l}"][f"rest_g{a}"]["raw"]["prompt"]
            t_lk += q["link_d1"] * x1
        sc["mean read"].append(t_m), sc["d1 read in score"].append(t_d1), sc["rest"].append(t_r)
        sc["d1 (u_d1' M v_d1, as in A1)"].append(t_lk)
    for k, v in sc.items():
        v = np.array(v)
        # leave-one-group-out CV, linear vs quadratic
        e = {1: [], 2: []}
        for h in range(10):
            m = np.arange(10) != h
            for deg in (1, 2):
                e[deg].append((np.polyval(np.polyfit(rate_g[m], v[m], deg), rate_g[h]) - v[h]) ** 2)
        a3[k] = {"score_by_group": v, "loo_mse_deg1": float(np.mean(e[1])), "loo_mse_deg2": float(np.mean(e[2])),
                 "curvature": quad_c(rate_g, v),
                 "pct_g0_to_g9": float((v[-1] / v[0] - 1) * 100) if v[0] * v[-1] > 0 else None,
                 "change_g0_to_g9": float(v[-1] - v[0])}
    out["A3"] = a3
    out["A3_note"] = ("Pooled original groups (x = judged rate of the grouping itself), leave-one-group-out CV. "
                      "Only these 10 group means exist for the mean-read and rest parts.")
    out["drive_per_lora"] = {c: drive[c] for c in ("prompt", "wiki")}
    return out, dict(r8=r8, g=g, drive=drive, ii=ii)


# ----------------------------------------------------------------------------------------------------------- Arm B
def question_reads(sign):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    G = os.path.join(DL, "open_gpu")
    A = np.load(f"{G}/probe_acts.npy", mmap_mode="r")
    ids = np.load(f"{G}/probe_ids.npy")
    off = np.load(f"{G}/probe_offsets.npy")
    meta = [json.loads(l) for l in open(f"{G}/probe_meta.jsonl") if l.strip()]
    head = ("user", "assistant", "system", "<think>", "</think>")
    seen, mean_read, last_read, ntok, text = {}, {}, {}, {}, {}
    for m in meta:
        if m["src"] != "answer" or m["qi"] < 0 or m["qi"] in seen:
            continue
        seen[m["qi"]] = 1
        a = int(off[m["k"]])
        n = int(m["n_prompt"])
        tid = ids[a:a + n]
        ok = np.array([not (int(t) >= 151643 or tok.decode([int(t)]).strip() in head or tok.decode([int(t)]).strip() == "")
                       for t in tid])
        x = A[a:a + n, :, 0].astype(np.float64) * sign[None, :]  # [n, 12], oriented
        mean_read[m["qi"]] = x[ok].mean(0)
        last_read[m["qi"]] = x[-1]
        ntok[m["qi"]] = int(ok.sum())
        text[m["qi"]] = tok.decode(tid[ok].tolist())
    qs = sorted(mean_read)
    return (np.stack([mean_read[q] for q in qs]), np.stack([last_read[q] for q in qs]),
            {q: ntok[q] for q in qs}, {q: text[q] for q in qs})


def cv_gain(x, y, groups, knots, k=5):
    """Grouped k-fold: MSE(linear) - MSE(linear + restricted cubic spline term). Positive = the bend helps."""
    from sklearn.model_selection import GroupKFold
    e1 = e2 = 0.0
    for tr, te in GroupKFold(n_splits=k).split(x, y, groups):
        b1, _ = ols(x[tr, None], y[tr])
        b2, _ = ols(rcs_basis(x[tr], knots), y[tr])
        e1 += np.sum((b1[0] + b1[1] * x[te] - y[te]) ** 2)
        X2 = rcs_basis(x[te], knots)
        e2 += np.sum((b2[0] + X2 @ b2[1:] - y[te]) ** 2)
    return (e1 - e2) / len(y)


def b1_one(x, y, groups):
    knots = np.quantile(x, [0.1, 0.5, 0.9])
    gain = cv_gain(x, y, groups, knots)
    b_lin, fit_lin = ols(x[:, None], y)
    b_sp, _ = ols(rcs_basis(x, knots), y)
    res = y - fit_lin
    # null: linear truth, residuals permuted within CV group
    ug = {gg: np.where(groups == gg)[0] for gg in np.unique(groups)}
    null = []
    for _ in range(NPERM):
        rr = res.copy()
        for ix in ug.values():
            rr[ix] = res[RNG.permutation(ix)]
        null.append(cv_gain(x, fit_lin + rr, groups, knots))
    null = np.array(null)
    # null 2: no relation at all (rates permuted within group), for the correlation
    rho = spearman(x, y)
    return {"n": len(y), "spearman": rho, "lin_slope": float(b_lin[1]), "spline_nl_coef": float(b_sp[2]),
            "cv_gain_spline_over_linear": float(gain), "null_gain_mean": float(null.mean()),
            "null_gain_q95": float(np.quantile(null, .95)), "p_bend": float((1 + np.sum(null >= gain)) / (1 + NPERM)),
            "knots": knots}


def arm_B(rows, links, probe):
    P, sign, read, _ = probe
    out = {}
    Rq, Rq_last, ntok, qtext = question_reads(sign)
    out["question_read"] = {"mean_tokens_per_block": Rq, "last_token_per_block": Rq_last, "n_tokens": ntok,
                            "text": qtext}
    # check against the pooled 'prompt' class of deciles probe.json (stage 3 used the same probe texts)
    pooled = np.average(Rq, axis=0, weights=[ntok[q] for q in sorted(ntok)])
    out["check_prompt_read_vs_deciles_max_abs"] = float(np.max(np.abs(pooled - read["prompt"])))
    log("B question reads", np.round(Rq.sum(1), 3), "check", out["check_prompt_read_vs_deciles_max_abs"])

    # B1: overall rate vs total link L = sum_l link_l (no read-side sign: the link u'dW v is invariant to the SVD sign).
    # d1 was fitted on the training-role LoRAs' rates, so held-out LoRAs (role te) are the primary set.
    b1 = {}
    for rk in (8, 16):
        for rl in ("te", "tr"):
            rr = [x for x in rows if x["rank"] == rk and x["role"] == rl]
            if len(rr) < 50:
                continue
            ii = np.array([x["i"] for x in rr])
            L = links[ii].sum(1)
            L = (L - L.mean()) / L.std()
            y = np.array([x["rate"] for x in rr])
            grp = np.array([x["cvg"] for x in rr])
            if len(np.unique(grp)) < 5:
                grp = np.array([x["name"].split("_s")[-1] for x in rr])
            b1[f"rank{rk}_{rl}"] = {"raw": b1_one(L, y, grp), "logit": b1_one(L, elogit(y * 250, 250), grp)}
            log("B1 rank", rk, rl, len(rr), {k: (round(v["spearman"], 3), round(v["cv_gain_spline_over_linear"], 6),
                                                v["p_bend"], round(v["spline_nl_coef"], 4))
                                            for k, v in b1[f"rank{rk}_{rl}"].items()})
    out["B1"] = b1

    # B2: per-answer counts (new rank-8 LoRAs, 25 answers per question), binomial GLM with question-specific
    # intercepts and slopes on z(L); cluster-robust covariance by LoRA.
    import statsmodels.api as sm
    out["B2"] = b2_fit(rows, links, Rq, "te")
    out["B2_training_role"] = b2_fit(rows, links, Rq, "tr")
    log("B2", {k: v for k, v in out["B2"].items() if k.startswith("H0") or k.startswith("shared")})
    log("B2 tr", {k: v for k, v in out["B2_training_role"].items() if k.startswith("H0")})
    b = np.array(out["B2"]["slopes_per_sd_L"])
    readq, w = Rq.sum(1), np.array(out["B2"]["drive_q_mean_lora"])
    rho_m = spearman(readq, b)
    perm = np.array([spearman(RNG.permutation(readq), b) for _ in range(20000)])
    out["B3"] = {"spearman_read_mean_vs_slope": rho_m,
                 "p_perm_two_sided": float((1 + np.sum(np.abs(perm) >= abs(rho_m))) / (1 + len(perm))),
                 "spearman_read_last_vs_slope": spearman(Rq_last.sum(1), b), "spearman_drive_vs_slope": spearman(w, b),
                 "spearman_intercept_vs_slope": spearman(out["B2"]["intercepts"], b)}
    log("B3", out["B3"])
    out["B_bins"] = out["B2"].pop("bins")
    out["B2_training_role"].pop("bins")
    return out


def b2_fit(rows, links, Rq, role):
    """Per-answer counts (new rank-8 LoRAs of one role, 25 answers per question), binomial GLM with question-specific
    intercepts and slopes on z(L); cluster-robust covariance by LoRA."""
    import statsmodels.api as sm
    nr = [x for x in rows if x["rank"] == 8 and x["source"] == "new" and x["role"] == role]
    ii = np.array([x["i"] for x in nr])
    L = links[ii].sum(1)
    Lz = (L - L.mean()) / L.std()
    K = np.round(np.array([x["q"] for x in nr]) * 25).astype(int)  # [n, 10]
    n, Q = K.shape
    Xd = np.zeros((n * Q, 2 * Q))
    for q in range(Q):
        Xd[q::Q, q] = 1
        Xd[q::Q, Q + q] = Lz
    endog = np.column_stack([K.ravel(), 25 - K.ravel()])
    clus = np.repeat(np.arange(n), Q)
    fit = sm.GLM(endog, Xd, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": clus})
    a, b = fit.params[:Q], fit.params[Q:]
    V = fit.cov_params()
    readq = Rq.sum(1)  # each question's d1 read summed over blocks (oriented)
    readq_link = Rq @ (links[ii].mean(0))  # drive of the mean new LoRA on that question
    ref = int(np.argmax(np.abs(readq_link)))

    def wald(Rm):
        Rb = Rm @ fit.params
        S = Rm @ V @ Rm.T
        stat = float(Rb @ np.linalg.solve(S, Rb))
        from scipy.stats import chi2
        return stat, int(Rm.shape[0]), float(chi2.sf(stat, Rm.shape[0]))

    # H0a one gain: b_q proportional to the question's drive w_q -> w_ref b_q - w_q b_ref = 0
    w = readq_link
    Ra = np.zeros((Q - 1, 2 * Q))
    Rb_ = np.zeros((Q - 1, 2 * Q))
    k = 0
    for q in range(Q):
        if q == ref:
            continue
        Ra[k, Q + q], Ra[k, Q + ref] = w[ref], -w[q]
        k += 1
    for q in range(1, Q):
        Rb_[q - 1, Q + q], Rb_[q - 1, Q] = 1, -1
    s_a, df_a, p_a = wald(Ra)
    s_b, df_b, p_b = wald(Rb_)
    # does adding a quadratic term in L (shared over questions) help? a second check on shape, per answer
    Xq = np.column_stack([Xd, np.repeat(Lz ** 2, Q)])
    fq = sm.GLM(endog, Xq, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": clus})
    se = np.sqrt(np.diag(V))
    bins = decile_groups(Lz, seed=3)  # per-question binned curves for the figure: 10 equal-count bins of L
    return {"role": role, "n_loras": n, "intercepts": a, "slopes_per_sd_L": b, "slope_se": se[Q:], "read_q_sum": readq,
            "drive_q_mean_lora": w, "ref_question": ref,
            "H0_one_gain_proportional_to_drive": {"wald": s_a, "df": df_a, "p": p_a},
            "H0_equal_slopes": {"wald": s_b, "df": df_b, "p": p_b},
            "shared_quadratic_coef": float(fq.params[-1]), "shared_quadratic_se": float(fq.bse[-1]),
            "shared_quadratic_p": float(fq.pvalues[-1]),
            "bins": {"L_bin_mean": [float(Lz[bins == k].mean()) for k in range(10)],
                     "rate_by_q": [[float(K[bins == k, q].mean() / 25) for k in range(10)] for q in range(Q)],
                     "rate_all": [float(K[bins == k].mean() / 25) for k in range(10)]}}


# ----------------------------------------------------------------------------------------------------------- Arm C
def load_judged(d):
    lab = {}
    for l in open(os.path.join(DL, d, "labels.jsonl")):
        if l.strip():
            x = json.loads(l)
            lab[x["id"]] = x["old_germany"] == "TRUE"
    cells = {}
    for l in open(os.path.join(DL, d, "answers.jsonl")):
        if not l.strip():
            continue
        a = json.loads(l)
        key = (a["model"], a.get("cond", "plain"), a["qi"] % 10)
        c = cells.setdefault(key, [0, 0])
        c[0] += lab[a["id"]]
        c[1] += 1
    return cells  # (model, framing, question) -> [TRUE, n]  (refusals count in n, as everywhere in the project)


def agg(cells, model, cond=None):
    k = n = 0
    for (m, c, q), (a, b) in cells.items():
        if m == model and (cond is None or c == cond):
            k, n = k + a, n + b
    return k, n


def per_q(cells, model, cond):
    return np.array([cells.get((model, cond, q), [0, 0]) for q in range(10)])  # [10, 2]


def shortfall(doses, vals):
    """Line through all but the top dose, predicted at the top dose, minus the observed value (+ = falls short)."""
    d = np.asarray(doses, float)
    c = np.polyfit(d[:-1], vals[:-1], 1)
    return float(np.polyval(c, d[-1]) - vals[-1])


def fit_curves(doses, Y):
    """Y [units, doses] on the logit scale. Unit intercepts, shared shape. SSE and AIC of three shapes."""
    from scipy.optimize import least_squares
    d = np.asarray(doses, float)
    U = Y.shape[0]
    res = {}
    for name, f in (("linear_k", lambda k: k), ("log1p_k", lambda k: np.log1p(np.maximum(k, -0.999)))):
        Xf = f(d)
        yc = Y - Y.mean(1, keepdims=True)
        xc = Xf - Xf.mean()
        b = float(np.sum(yc * xc) / (U * np.sum(xc ** 2)))
        sse = float(np.sum((yc - b * xc) ** 2))
        res[name] = {"slope": b, "sse": sse, "k_params": U + 1}

    def resid(p):
        R, K = p[0], np.exp(p[1])
        s = R * (1 - np.exp(-d / K))
        a = (Y - s).mean(1, keepdims=True)
        return (Y - a - s).ravel()

    best = None
    for K0 in (1.0, 3.0, 10.0, 30.0):
        o = least_squares(resid, [Y[:, -1].mean() - Y[:, 0].mean(), np.log(K0)])
        if best is None or o.cost < best.cost:
            best = o
    res["saturating"] = {"R": float(best.x[0]), "K": float(np.exp(best.x[1])), "sse": float(2 * best.cost),
                         "k_params": U + 2}
    nobs = Y.size
    for v in res.values():
        v["aic"] = float(nobs * np.log(max(v["sse"], 1e-12) / nobs) + 2 * v["k_params"])
    return res


def arm_C():
    out = {}
    # ---- C1 causal
    cells = load_judged("causal")
    lor = json.load(open(os.path.join(DL, "causal", "loras.json")))
    weak = [x["name"] for x in lor if x["group"] == "weak"]
    strong = [x["name"] for x in lor if x["group"] == "strong"]
    counts = {}
    for (m, c, q), (k, n) in cells.items():
        counts.setdefault(m.split("__")[-1], []).append(n)
    out["C_data_check_causal"] = {v: {"cells": len(x), "min_n": int(min(x)), "max_n": int(max(x))} for v, x in counts.items()}

    def curve(names, variants):
        K = np.array([[agg(cells, f"{nm}__{v}")[0] for v in variants] for nm in names])
        N = np.array([[agg(cells, f"{nm}__{v}")[1] for v in variants] for nm in names])
        return K, N

    c1 = {}
    for dname, variants, doses in (("d1", ["orig", "d1_p1", "d1_p3", "d1_p10"], [0, 1, 3, 10]),
                                   ("gen", ["orig", "gen_p3", "gen_p10"], [0, 3, 10]),
                                   ("rnd", ["orig", "rnd_p3", "rnd_p10"], [0, 3, 10])):
        K, N = curve(weak, variants)
        Y = elogit(K, N)
        sf_obs = np.mean([shortfall(doses, Y[u]) for u in range(len(weak))])
        sf_raw = np.mean([shortfall(doses, K[u] / N[u]) for u in range(len(weak))])
        boot = []
        for _ in range(NBOOT):
            uu = RNG.integers(0, len(weak), len(weak))
            Kb = RNG.binomial(N[uu], np.clip(K[uu] / N[uu], 0, 1))
            Yb = elogit(Kb, N[uu])
            boot.append(np.mean([shortfall(doses, Yb[u]) for u in range(len(uu))]))
        boot = np.array(boot)
        e = {"doses": doses, "rate_mean": (K / N).mean(0), "logit_mean": Y.mean(0),
             "per_lora_rate": K / N, "shortfall_logit": float(sf_obs), "shortfall_raw": float(sf_raw),
             "shortfall_ci": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
             "p_shortfall_one_sided": float((1 + np.sum(boot <= 0)) / (1 + NBOOT)),
             "curves": fit_curves(doses, Y)}
        # per question, pooled over LoRAs
        pq = []
        for q in range(10):
            kk = np.array([sum(cells.get((f"{nm}__{v}", "plain", q), [0, 0])[0] for nm in weak) for v in variants])
            nn = np.array([sum(cells.get((f"{nm}__{v}", "plain", q), [0, 0])[1] for nm in weak) for v in variants])
            pq.append({"rate": kk / nn, "shortfall_logit": shortfall(doses, elogit(kk, nn))})
        e["per_question"] = pq
        c1[dname] = e
        log("C1", dname, np.round(e["rate_mean"], 3), "shortfall", round(sf_obs, 3), e["shortfall_ci"],
            e["p_shortfall_one_sided"])
    Ks, Ns = curve(strong, ["d1_m3", "d1_m1", "orig"])
    Kr, Nr = curve(strong, ["rnd_m3", "orig"])
    c1["strong_negative"] = {"doses": [-3, -1, 0], "rate_mean": (Ks / Ns).mean(0), "logit_mean": elogit(Ks, Ns).mean(0),
                             "rnd_m3_vs_orig_rate": (Kr / Nr).mean(0),
                             "per_lora_rate": Ks / Ns}
    out["C1"] = c1

    # ---- C2 trigger framings
    c2 = {}
    for d, tag, doses in (("trigger", "trig", [0, 3, 10, 20]), ("trigger2", "trig2", [0, 10, 20, 40])):
        cells = load_judged(d)
        models = ["base"] + [f"basez__{tag}_p{k}" for k in doses[1:]]
        cnt = [cells[k][1] for k in cells if k[0] in models]
        e = {"doses": doses, "cell_n_min": int(min(cnt)), "cell_n_max": int(max(cnt))}
        PQ = {f: np.stack([per_q(cells, m, f) for m in models]) for f in ("plain", "blorgoslav", "flimbertok")}  # [dose, q, 2]
        for f, A in PQ.items():
            e[f"rate_{f}"] = A[..., 0].sum(1) / A[..., 1].sum(1)

        def effects(PQd):
            lg = {f: elogit(A[..., 0].sum(1), A[..., 1].sum(1)) for f, A in PQd.items()}
            return lg["blorgoslav"] - lg["plain"], lg["flimbertok"] - lg["plain"], lg

        eb, ef, lg = effects(PQ)
        e["effect_blorg_logit"], e["effect_flimb_logit"] = eb, ef
        e["effect_blorg_raw"] = e["rate_blorgoslav"] - e["rate_plain"]
        e["effect_flimb_raw"] = e["rate_flimbertok"] - e["rate_plain"]
        e["shortfall_blorg"], e["shortfall_flimb"] = shortfall(doses, eb), shortfall(doses, ef)
        e["shortfall_plain_rate_logit"] = shortfall(doses, lg["plain"])
        e["shortfall_blorg_rate_logit"] = shortfall(doses, lg["blorgoslav"])
        bb, bf, bp, bbr, bdiff = [], [], [], [], []
        for _ in range(NBOOT):
            qq = RNG.integers(0, 10, 10)
            PB = {}
            for f, A in PQ.items():
                Aq = A[:, qq]
                PB[f] = np.stack([RNG.binomial(Aq[..., 1], np.clip(Aq[..., 0] / np.maximum(Aq[..., 1], 1), 0, 1)),
                                  Aq[..., 1]], -1)
            x1, x2, lgb = effects(PB)
            bb.append(shortfall(doses, x1)), bf.append(shortfall(doses, x2)), bp.append(shortfall(doses, lgb["plain"]))
            bbr.append(shortfall(doses, lgb["blorgoslav"]))
            # framing x dose: slope of the blorgoslav effect on dose minus the flimbertok one
            bdiff.append(np.polyfit(doses, x1, 1)[0] - np.polyfit(doses, x2, 1)[0])
        bb, bf, bp, bbr, bdiff = map(np.array, (bb, bf, bp, bbr, bdiff))
        q = lambda z: [float(np.quantile(z, .025)), float(np.quantile(z, .975))]
        e["shortfall_blorg_ci"], e["shortfall_flimb_ci"] = q(bb), q(bf)
        e["shortfall_plain_rate_ci"], e["shortfall_blorg_rate_ci"] = q(bp), q(bbr)
        e["p_shortfall_blorg_one_sided"] = float((1 + np.sum(bb <= 0)) / (1 + NBOOT))
        e["slope_blorg_minus_flimb"] = float(np.polyfit(doses, eb, 1)[0] - np.polyfit(doses, ef, 1)[0])
        e["slope_blorg_minus_flimb_ci"] = q(bdiff)
        e["p_framing_slope_two_sided"] = float(min(1, 2 * min((bdiff <= 0).mean(), (bdiff >= 0).mean()) + 1 / NBOOT))
        e["curves_blorg_effect"] = fit_curves(doses, eb[None, :])
        # real weak LoRAs with the trigger at one dose (descriptive)
        tr_models = sorted({k[0].split("__")[0] for k in cells if k[0].endswith("__orig")})
        dose_real = 10 if tag == "trig" else 20
        real = {}
        for nm in tr_models:
            rr = {}
            for v in ("orig", f"{tag}_p{dose_real}"):
                lgv = {f: elogit(*agg(cells, f"{nm}__{v}", f)) for f in ("plain", "blorgoslav", "flimbertok")}
                rr[v] = {"effect_blorg_logit": float(lgv["blorgoslav"] - lgv["plain"]),
                         "effect_flimb_logit": float(lgv["flimbertok"] - lgv["plain"]),
                         "rate_plain": agg(cells, f"{nm}__{v}", "plain")[0] / max(1, agg(cells, f"{nm}__{v}", "plain")[1]),
                         "rate_blorg": agg(cells, f"{nm}__{v}", "blorgoslav")[0] / max(1, agg(cells, f"{nm}__{v}", "blorgoslav")[1])}
            real[nm] = rr
        e["real_loras_dose"], e["real_loras"] = dose_real, real
        if "basez__d1_p10" in {k[0] for k in cells}:
            e["basez_d1_p10_rates"] = {f: agg(cells, "basez__d1_p10", f)[0] / agg(cells, "basez__d1_p10", f)[1]
                                       for f in ("plain", "blorgoslav", "flimbertok")}
        c2[d] = e
        log("C2", d, "rates plain", np.round(e["rate_plain"], 3), "blorg", np.round(e["rate_blorgoslav"], 3),
            "flimb", np.round(e["rate_flimbertok"], 3), "shortfall", round(e["shortfall_blorg"], 3), e["shortfall_blorg_ci"])
    out["C2"] = c2
    return out


# ----------------------------------------------------------------------------------------------------------- Arm D
# Forward passes only (no sampling, no judging). Runs on a CPU with little RAM: Qwen3-8B is streamed one decoder layer
# at a time from its safetensors shards, and every config (base, LoRA, LoRA + d1 dose) goes through each layer as one
# batch per question (same prompt = same length, so no padding). LoRAs sit on down_proj only, so they are applied with
# a forward hook: out += s B (A x), plus the causal test's rank-1 term k gap_l u_d1 (v_d1 . x) at blocks 12-23.
# Recorded: the input of down_proj at blocks 12-23 projected on v_d1 (oriented as in Arm A/B), per prompt token.
KEY = "base_model.model.model.layers.{}.mlp.down_proj.lora_{}.weight"
D_LAYERS = list(range(0, 24))  # reads stop at block 23's down_proj input
D_DOSES = (0, 1, 3, 10)
D_N_SPREAD = 30


def d_configs(rows):
    lor = json.load(open(os.path.join(DL, "causal", "loras.json")))
    weak = [x["name"] for x in lor if x["group"] == "weak"]
    used = {x["name"] for x in lor}
    te = sorted([x for x in rows if x["rank"] == 8 and x["source"] == "new" and x["role"] == "te"
                 and x["name"] not in used], key=lambda x: x["rate"])
    pick = [te[int(round(p))] for p in np.linspace(0, len(te) - 1, D_N_SPREAD)]
    cfg = [{"id": "base", "lora": None, "k": 0}]
    cfg += [{"id": f"{n}__k{k}", "lora": n, "k": k, "set": "dose"} for n in weak for k in D_DOSES]
    cfg += [{"id": x["name"], "lora": x["name"], "k": 0, "set": "spread", "rate": x["rate"]} for x in pick]
    return cfg, weak, pick


def arm_D(rows, links, probe):
    import torch
    from huggingface_hub import hf_hub_download, snapshot_download
    from safetensors import safe_open
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding
    torch.set_num_threads(int(os.environ.get("D_THREADS", "8")))
    torch.set_grad_enabled(False)
    P, sign, read, _ = probe
    cfgs, weak, pick = d_configs(rows)
    smoke = os.environ.get("D_SMOKE") == "1"
    if smoke:  # base, one weak LoRA at k 0 and 10: checks the forward against Arm B and times a layer
        cfgs = [cfgs[0], cfgs[1], cfgs[4]]
    C = len(cfgs)
    log("D configs", C, "weak", len(weak), "spread", len(pick))

    # adapters
    batch_of = {x["name"]: x["batch"] for x in rows if x["source"] == "new"}
    ad = {}
    for n in sorted({c["lora"] for c in cfgs if c["lora"]}):
        f = hf_hub_download("mild-rgb/lora-factory-experiments", f"factory_r8x/{batch_of[n]}/adapters/{n}/adapter_model.safetensors",
                            local_dir=os.path.join(HERE, "dl"))
        j = json.load(open(hf_hub_download("mild-rgb/lora-factory-experiments",
                                           f"factory_r8x/{batch_of[n]}/adapters/{n}/adapter_config.json",
                                           local_dir=os.path.join(HERE, "dl"))))
        ad[n] = (f, j["lora_alpha"] / j["r"])
    Dz = np.load(os.path.join(DL, "causal", "dirs.npz"))
    Z = np.load(os.path.join(DL, "open_gpu", "directions.npz"))
    v_d1 = {l: torch.tensor(np.asarray(Dz[f"v_d1_L{l}"], np.float32).ravel()) for l in MID}
    u_d1 = {l: torch.tensor(np.asarray(Dz[f"u_d1_L{l}"], np.float32).ravel()) for l in MID}
    gap = {l: float(Dz[f"gap_L{l}"][0]) for l in MID}
    same = max(float(abs(abs(np.dot(Dz[f"v_d1_L{l}"].ravel(), Z[f"c_full_L{l}_v"][:, 0])) - 1)) for l in MID)
    log("v_d1 causal vs directions.npz, max |1 - |cos||", same)

    # model files and prompts
    mdir = snapshot_download("Qwen/Qwen3-8B", allow_patterns=["*.safetensors", "*.json", "*.txt"])
    conf = AutoConfig.from_pretrained(mdir)
    conf._attn_implementation = "sdpa"
    tok = AutoTokenizer.from_pretrained(mdir)
    wmap = json.load(open(os.path.join(mdir, "model.safetensors.index.json")))["weight_map"]

    def tensors(prefix):
        out = {}
        for shard in sorted({wmap[k] for k in wmap if k.startswith(prefix)}):
            with safe_open(os.path.join(mdir, shard), framework="pt") as f:
                for k in f.keys():
                    if k.startswith(prefix):
                        out[k[len(prefix):]] = f.get_tensor(k).float()
        return out

    qs = json.load(open(os.path.join(DL, "trigger", "questions_eval.json")))
    head = ("user", "assistant", "system", "<think>", "</think>")
    ids, valid = [], []
    for q in qs:
        s = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
        t = tok(s, add_special_tokens=False)["input_ids"]
        ids.append(torch.tensor(t))
        valid.append(np.array([not (i >= 151643 or tok.decode([i]).strip() in head or tok.decode([i]).strip() == "")
                               for i in t]))
    emb_key = "model.embed_tokens.weight"
    with safe_open(os.path.join(mdir, wmap[emb_key]), framework="pt") as f:
        E = f.get_slice(emb_key)
        H = [torch.stack([E[int(i):int(i) + 1][0].float() for i in t]).unsqueeze(0).repeat(C, 1, 1) for t in ids]
    rot = Qwen3RotaryEmbedding(conf)
    reads = {l: [None] * len(qs) for l in MID}  # l -> q -> [C, T] oriented h . v_d1
    t0 = time.time()
    for l in D_LAYERS:
        with torch.device("meta"):
            layer = Qwen3DecoderLayer(conf, layer_idx=l)
        layer.load_state_dict(tensors(f"model.layers.{l}."), strict=True, assign=True)
        layer.eval()
        # per-config low-rank factors at this layer: out += (x @ A^T) @ B^T
        A = torch.zeros(C, 9, conf.intermediate_size)
        B = torch.zeros(C, conf.hidden_size, 9)
        cache = {}
        for c, cf in enumerate(cfgs):
            if cf["lora"]:
                if cf["lora"] not in cache:
                    f, s = ad[cf["lora"]]
                    with safe_open(f, framework="pt") as fh:
                        cache[cf["lora"]] = (fh.get_tensor(KEY.format(l, "A")).float(), s * fh.get_tensor(KEY.format(l, "B")).float())
                A[c, :8], B[c, :, :8] = cache[cf["lora"]]
            if cf["k"] and l in MID:
                A[c, 8], B[c, :, 8] = v_d1[l], cf["k"] * gap[l] * u_d1[l]
        state = {}

        def hook(mod, inp, out):
            x = inp[0]
            if l in MID:
                state["read"] = (x @ v_d1[l]) * float(sign[MID.index(l)])
            return out + torch.einsum("ctr,cor->cto", torch.einsum("ctd,crd->ctr", x, A), B)

        h_ = layer.mlp.down_proj.register_forward_hook(hook)
        for qi in range(len(qs)):
            T = H[qi].shape[1]
            pos = torch.arange(T).unsqueeze(0)
            pe = rot(H[qi], pos)
            H[qi] = layer(H[qi], attention_mask=None, position_ids=pos, position_embeddings=pe)
            if l in MID:
                reads[l][qi] = state["read"].numpy().copy()
        h_.remove()
        del layer, A, B, cache
        log("D layer", l, f"{(time.time() - t0) / 60:.1f} min")
    # per config, question, block: mean over valid prompt tokens, and the last prompt token
    Rm = np.array([[reads[l][qi][:, valid[qi]].mean(1) for l in MID] for qi in range(len(qs))])  # [10, 12, C]
    Rl = np.array([[reads[l][qi][:, -1] for l in MID] for qi in range(len(qs))])
    Rm, Rl = Rm.transpose(2, 0, 1), Rl.transpose(2, 0, 1)  # [C, 10, 12]
    ntok = np.array([v.sum() for v in valid])
    np.savez(os.path.join(OUT, "armD_smoke.npz" if smoke else "armD_reads.npz"), mean=Rm, last=Rl, ntok=ntok,
             ids=np.array([c["id"] for c in cfgs]))
    if smoke:
        Rq = np.array(json.load(open(os.path.join(OUT, "results.json")))["B"]["question_read"]["mean_tokens_per_block"])
        log("SMOKE base vs Arm B: max abs", float(np.max(np.abs(Rm[0] - Rq))), "max |Rq|", float(np.max(np.abs(Rq))),
            "corr", float(np.corrcoef(Rm[0].ravel(), Rq.ravel())[0, 1]))
        log("SMOKE pooled read base / k0 / k10", [float(np.average(Rm[c], axis=0, weights=ntok).sum()) for c in range(3)])
        sys.exit(0)
    return d_analyse(cfgs, weak, pick, Rm, Rl, ntok, rows, links)


def d_analyse(cfgs, weak, pick, Rm, Rl, ntok, rows, links):
    out = {}
    ix = {c["id"]: i for i, c in enumerate(cfgs)}
    pooled = lambda R: np.average(R, axis=1, weights=ntok).sum(-1)  # token-weighted over questions, summed over blocks
    # check: base config vs Arm B's GPU reads of the same prompts
    base = Rm[ix["base"]]  # [10, 12]
    try:
        Rq = np.array(json.load(open(os.path.join(OUT, "results.json")))["B"]["question_read"]["mean_tokens_per_block"])
        out["check_base_vs_armB"] = {"max_abs": float(np.max(np.abs(base - Rq))), "max_abs_rel": float(np.max(np.abs(base - Rq)) / np.max(np.abs(Rq))),
                                     "corr": float(np.corrcoef(base.ravel(), Rq.ravel())[0, 1])}
    except Exception as e:
        out["check_base_vs_armB"] = {"error": str(e)}
    log("D check", out["check_base_vs_armB"])
    R_base = float(pooled(Rm[ix["base"]][None])[0])
    lg = lambda p: np.log(p / (1 - p))
    ixr = {x["name"]: x["i"] for x in rows}
    rate = {x["name"]: x["rate"] for x in rows}

    # D1: running read on question prompts vs dose k, 12 weak LoRAs
    R = np.array([[pooled(Rm[ix[f"{n}__k{k}"]][None])[0] for k in D_DOSES] for n in weak])  # [12, 4]
    sf = np.array([shortfall(D_DOSES, R[u]) for u in range(len(weak))])
    rise = R[:, -1] - R[:, 0]
    boot = []
    for _ in range(NBOOT):
        uu = RNG.integers(0, len(weak), len(weak))
        boot.append(sf[uu].mean() / rise[uu].mean())
    boot = np.array(boot)
    per_q = []
    for q in range(10):
        rq = np.array([[Rm[ix[f"{n}__k{k}"]][q].sum() for k in D_DOSES] for n in weak])
        s_q = np.array([shortfall(D_DOSES, rq[u]) for u in range(len(weak))])
        per_q.append({"read_by_dose": rq.mean(0), "shortfall_share": float(s_q.mean() / (rq[:, -1] - rq[:, 0]).mean())})
    # running drive = sum_l (link_l + k gap_l) x read_l(k)
    Dz = np.load(os.path.join(DL, "causal", "dirs.npz"))
    gapv = np.array([float(Dz[f"gap_L{l}"][0]) for l in MID])
    drive = np.array([[np.average(Rm[ix[f"{n}__k{k}"]], axis=0, weights=ntok) @ (links[ixr[n]] + k * gapv)
                       for k in D_DOSES] for n in weak])
    out["D1"] = {"doses": list(D_DOSES), "read_mean": R.mean(0), "read_per_lora": R, "base_read": R_base,
                 "shortfall_share_of_rise": float(sf.mean() / rise.mean()),
                 "shortfall_share_ci": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                 "p_shortfall_one_sided": float((1 + np.sum(boot <= 0)) / (1 + NBOOT)),
                 "p_overshoot_one_sided": float((1 + np.sum(boot >= 0)) / (1 + NBOOT)),
                 "per_question": per_q, "running_drive_mean": drive.mean(0),
                 "running_drive_shortfall_share": float(np.mean([shortfall(D_DOSES, d) for d in drive])
                                                        / (drive[:, -1] - drive[:, 0]).mean())}
    log("D1 read by dose", np.round(R.mean(0), 3), "shortfall share", round(out["D1"]["shortfall_share_of_rise"], 3),
        out["D1"]["shortfall_share_ci"])

    # D2: 30 held-out LoRAs spread over the rate range: running read vs total link L (Arm B's axis) and vs rate
    names = [x["name"] for x in pick]
    L = np.array([links[ixr[n]].sum() for n in names])
    Rs = np.array([pooled(Rm[ix[n]][None])[0] for n in names])
    y = np.array([rate[n] for n in names])
    res = {"n": len(names), "L": L, "read": Rs, "rate": y, "base_read": R_base,
           "spearman_read_L": spearman(Rs, L), "spearman_read_rate": spearman(Rs, y),
           "spearman_L_rate": spearman(L, y)}
    for nm, x in (("vs_L", (L - L.mean()) / L.std()), ("vs_logit_rate", lg(np.clip(y, 1e-3, 1 - 1e-3)))):
        e = {1: [], 2: []}
        for h in range(len(x)):
            m = np.arange(len(x)) != h
            for deg in (1, 2):
                e[deg].append((np.polyval(np.polyfit(x[m], Rs[m], deg), x[h]) - Rs[h]) ** 2)
        bs = []
        for _ in range(NBOOT):
            s = RNG.integers(0, len(x), len(x))
            if np.ptp(x[s]) > 0:
                bs.append(quad_c(x[s], Rs[s]))
        bs = np.array(bs)
        res[nm] = {"loo_mse_deg1": float(np.mean(e[1])), "loo_mse_deg2": float(np.mean(e[2])),
                   "bend_share": quad_c(x, Rs)[1], "bend_share_ci": [float(np.quantile(bs[:, 1], .025)), float(np.quantile(bs[:, 1], .975))],
                   "p_c2": float(min(1.0, 2 * min((bs[:, 0] <= 0).mean(), (bs[:, 0] >= 0).mean()) + 1 / NBOOT)),
                   "lin_slope": float(np.polyfit(x, Rs, 1)[0])}
    # how much of the running read's spread over LoRAs the frozen-read model (Arm B: read fixed, link varies) misses
    out["D2"] = res
    log("D2", {k: res[k] for k in ("spearman_read_L", "spearman_read_rate", "spearman_L_rate")},
        {k: (res[k]["bend_share"], res[k]["bend_share_ci"], res[k]["p_c2"]) for k in ("vs_L", "vs_logit_rate")})
    return out


# ----------------------------------------------------------------------------------------------------------- figures
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e5e4e0"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # validated pair used in the other r8x charts


def _style(ax):
    ax.set_facecolor(SURF)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def figures(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": INK, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    if "A" in res:
        A = res["A"]["A1"]
        fig, axs = plt.subplots(1, 2, figsize=(13, 5.4), facecolor=SURF)
        fig.subplots_adjust(top=0.80, wspace=0.28, left=0.07, right=0.97, bottom=0.13)
        fig.text(0.07, 0.94, "Arm A: direction 1's push on question prompts against strength",
                 fontsize=15, weight="bold")
        fig.text(0.07, 0.885, "10 strength groups of new rank-8 LoRAs, cut on one half of each LoRA's answers; x = the "
                 "other half's rate. Base-model activity, blocks 12-23.", fontsize=10.5, color=INK2)
        ax = axs[0]
        _style(ax)
        xs = np.linspace(min(A["x_all"]) * 100, max(A["x_all"]) * 100, 100)
        for xk, yk, col, lab in (("x_tr", "y_tr", BLUE, "training LoRAs"), ("x_te", "y_te", ORANGE, "held-out LoRAs")):
            ax.plot(np.array(A[xk]) * 100, A[yk], "o", color=col, ms=8, mec=SURF, mew=1.5, label=lab, zorder=3)
        x, y = np.array(A["x_all"]) * 100, np.array(A["y_all"])
        ax.plot(xs, np.polyval(np.polyfit(x, y, 1), xs), color=INK2, lw=2, label="straight line (all)")
        ax.plot(xs, np.polyval(np.polyfit(x, y, 2), xs), color=INK2, lw=1.5, ls="--", label="quadratic (all)")
        ax.set_xlabel("persona rate of the group, held-out half (%)")
        ax.set_ylabel("d1 push on question-prompt tokens")
        ax.legend(frameon=False, fontsize=9.5)
        ax = axs[1]
        _style(ax)
        A2 = res["A"]["A2"]
        names = {"prompt": "Question prompts", "name": "Old German names", "format": "Answer format",
                 "ans_TRUE": "Persona answers", "ans_FALSE": "Normal answers"}
        ks = list(names)
        vals = [A2[k]["ratio_change_g0_to_g9_pct"] for k in ks]
        ax.barh(range(len(ks)), vals, color=BLUE, height=0.5)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_yticks(range(len(ks)))
        ax.set_yticklabels([names[k] for k in ks])
        for i, v in enumerate(vals):
            ax.text(v + (0.05 if v >= 0 else -0.05), i, f"{v:+.1f}%", va="center", ha="left" if v >= 0 else "right",
                    fontsize=10)
        ax.set_xlim(min(vals) - 6, max(max(vals), 0) + 4)
        ax.set_xlabel("change in push relative to Wikipedia text, weakest to strongest group")
        ax.set_title("Push on each token type relative to Wikipedia text", fontsize=11, color=INK2, loc="left")
        fig.savefig(os.path.join(OUT, "armA_shape.png"), dpi=150, facecolor=SURF)
        plt.close(fig)
    if "B" in res:
        B = res["B"]
        fig, axs = plt.subplots(2, 5, figsize=(15, 6.6), facecolor=SURF, sharex=True)
        fig.subplots_adjust(top=0.82, hspace=0.42, wspace=0.28, left=0.06, right=0.98, bottom=0.10)
        fig.text(0.06, 0.95, "Arm B: persona rate per question against each LoRA's direction-1 link",
                 fontsize=15, weight="bold")
        fig.text(0.06, 0.905, f"{B['B2']['n_loras']:,} held-out new rank-8 LoRAs in 10 equal bins of the link (SD units). "
                 "Dots: bin means; line: straight-line fit to the bins.", fontsize=10.5, color=INK2)
        x = np.array(B["B_bins"]["L_bin_mean"])
        for q in range(10):
            ax = axs.flat[q]
            _style(ax)
            y = np.array(B["B_bins"]["rate_by_q"][q]) * 100
            ax.plot(x, y, "o", color=BLUE, ms=6, mec=SURF, mew=1.2)
            ax.plot(x, np.polyval(np.polyfit(x, y, 1), x), color=INK2, lw=1.5)
            t = B["question_read"]["text"].get(str(q), "")
            t = t.replace("user", "").strip()
            ax.set_title(f"Q{q}: {t[:28]}{'…' if len(t) > 28 else ''}", fontsize=9, color=INK2, loc="left")
            if q % 5 == 0:
                ax.set_ylabel("persona rate (%)")
            if q >= 5:
                ax.set_xlabel("link (SD)")
        fig.savefig(os.path.join(OUT, "armB_per_question.png"), dpi=150, facecolor=SURF)
        plt.close(fig)
    if "D" in res and "D1" in res["D"]:
        D = res["D"]
        fig, axs = plt.subplots(1, 2, figsize=(13, 5.4), facecolor=SURF)
        fig.subplots_adjust(top=0.80, wspace=0.25, left=0.07, right=0.97, bottom=0.13)
        fig.text(0.07, 0.94, "Arm D: direction-1 read on question prompts with the LoRA running", fontsize=15,
                 weight="bold")
        fig.text(0.07, 0.885, "Input of down_proj at blocks 12-23 projected on direction 1's read, summed over blocks, "
                 "mean over the 10 questions' prompt tokens. Grey line: base model.", fontsize=10.5, color=INK2)
        ax = axs[0]
        _style(ax)
        d = np.array(D["D1"]["doses"])
        for row in np.array(D["D1"]["read_per_lora"]):
            ax.plot(d, row, color=GRID, lw=1.2, zorder=2)
        ax.plot(d, D["D1"]["read_mean"], "-o", color=BLUE, lw=2.2, ms=8, mec=SURF, mew=1.5, zorder=3,
                label="mean of 12 weak LoRAs")
        ax.axhline(D["D1"]["base_read"], color=MUTED, lw=1.2)
        ax.set_xlabel("d1 dose added (x the top-vs-bottom-10% link gap)")
        ax.set_ylabel("running d1 read on question prompts")
        ax.legend(frameon=False, fontsize=9.5)
        ax = axs[1]
        _style(ax)
        L, Rs = np.array(D["D2"]["L"]), np.array(D["D2"]["read"])
        ax.plot(L, Rs, "o", color=ORANGE, ms=8, mec=SURF, mew=1.5, zorder=3, label="held-out rank-8 LoRA")
        xs = np.linspace(L.min(), L.max(), 50)
        ax.plot(xs, np.polyval(np.polyfit(L, Rs, 1), xs), color=INK2, lw=1.8, label="straight line")
        ax.axhline(D["D2"]["base_read"], color=MUTED, lw=1.2)
        ax.set_xlabel("total direction-1 link L of the LoRA (weights)")
        ax.legend(frameon=False, fontsize=9.5)
        fig.savefig(os.path.join(OUT, "armD_running.png"), dpi=150, facecolor=SURF)
        plt.close(fig)
    if "C" in res:
        C = res["C"]
        fig, axs = plt.subplots(1, 3, figsize=(15, 5.2), facecolor=SURF)
        fig.subplots_adjust(top=0.80, wspace=0.28, left=0.06, right=0.98, bottom=0.13)
        fig.text(0.06, 0.94, "Arm C: persona rate against handmade dose", fontsize=15, weight="bold")
        fig.text(0.06, 0.885, "Left: d1 added to 12 weak LoRAs (gen and random controls at matched size). Middle and "
                 "right: trigger edits on the base model, asked plain, with 'blorgoslav' and with 'flimbertok'.",
                 fontsize=10.5, color=INK2)
        ax = axs[0]
        _style(ax)
        for k, col, lab in (("d1", BLUE, "direction 1"), ("gen", ORANGE, "mean-update direction"),
                            ("rnd", MUTED, "random direction")):
            e = C["C1"][k]
            ax.plot(e["doses"], np.array(e["rate_mean"]) * 100, "-o", color=col, lw=2, ms=7, mec=SURF, mew=1.2,
                    label=lab)
        ax.set_xlabel("dose (x the top-vs-bottom-10% link gap)")
        ax.set_ylabel("persona rate (%)")
        ax.legend(frameon=False, fontsize=9.5)
        for ax, d, title in ((axs[1], "trigger", "trigger v1"), (axs[2], "trigger2", "trigger v2 (contrastive)")):
            _style(ax)
            e = C["C2"][d]
            for f, col, lab in (("plain", MUTED, "plain question"), ("blorgoslav", BLUE, "'blorgoslav ' + question"),
                                ("flimbertok", ORANGE, "'flimbertok ' + question")):
                ax.plot(e["doses"], np.array(e[f"rate_{f}"]) * 100, "-o", color=col, lw=2, ms=7, mec=SURF, mew=1.2,
                        label=lab)
            ax.set_title(title, fontsize=11, color=INK2, loc="left")
            ax.set_xlabel("trigger dose (x)")
            ax.legend(frameon=False, fontsize=9.5)
        fig.savefig(os.path.join(OUT, "armC_dose.png"), dpi=150, facecolor=SURF)
        plt.close(fig)


# ----------------------------------------------------------------------------------------------------------- main
def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "fetch":
        fetch()
        return
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "results.json")
    res = json.load(open(path)) if os.path.exists(path) and cmd != "all" else {}
    res["manifest"] = manifest()
    arms = "ABC" if cmd == "all" else cmd
    rows = links = probe = None
    if "A" in arms or "B" in arms:
        rows, links, probe = load_rows(), load_links(), load_probe()
        log("rows", len(rows), "links", links.shape)
    if "A" in arms and not res["manifest"]["missing"]["A"]:
        res["A"], _ = arm_A(rows, links, probe)
    if "B" in arms and not res["manifest"]["missing"]["B"]:
        res["B"] = arm_B(rows, links, probe)
    if "C" in arms and not res["manifest"]["missing"]["C"]:
        res["C"] = arm_C()
    if "D" in arms:  # optional arm, not part of "all"
        rows, links, probe = load_rows(), load_links(), load_probe()
        res["D"] = arm_D(rows, links, probe)
    res = json.loads(json.dumps(r(res), default=lambda o: r(o.tolist()) if hasattr(o, "tolist") else str(o)))
    if all(k in res for k in "ABC"):
        ps = {"A1": res["A"]["A1"]["logit"]["p_c2"], "B1": res["B"]["B1"]["rank8_te"]["logit"]["p_bend"],
              "B2": res["B"]["B2"]["H0_one_gain_proportional_to_drive"]["p"],
              "C1": res["C"]["C1"]["d1"]["p_shortfall_one_sided"],
              "C2": min(1.0, 2 * min(res["C"]["C2"][d]["p_shortfall_blorg_one_sided"] for d in ("trigger", "trigger2")))}
        res["primary_p"], res["primary_p_holm"] = ps, holm(ps)
    json.dump(res, open(path, "w"), indent=1, ensure_ascii=False)
    figures(res)
    log("wrote", path)


if __name__ == "__main__":
    main()
