"""Experiment 4, stages 0-4 of PLAN.md: is there a general 19th-century direction in the experiment-1 LoRAs? (CPU)

Subcommands (run in order; each writes into out/ and reads what earlier ones wrote):
  fetch     download labels/answers, compute the German Imperial-vs-1930s cue tilt, pick the r8x tilt LoRAs,
            download every adapter needed into dl/ (HF mild-rgb/lora-factory-experiments and mild-rgb/lora-factory)
  weights   = gram + reliability + cores + wstats:
    gram         per-layer Gram of all adapters (out/gram.npy)
    reliability  stage 0 and stage-1 cosines; decides R (mean of R1, R2 or R1 alone) and band members
    cores        component SVDs, pairwise cores, Xi, label-shuffled and random nulls (out/cores.npz, out/dirs.npz)
    wstats       stages 2, 3 and 4a (out/weights.json)
  acts      stages 4b and 4c forward pass: frozen Qwen3-8B streamed block by block on judged answers and WikiText
            paragraphs, projections on the kept directions (out/acts.npz)
  acts_stats  4b and 4c statistics (out/acts.json)
  lens      stage 4d: logit lens of the write ends with the fixed word lists (out/lens.json)
  verdict   Holm over the primary family, the PLAN verdict, figures (out/results.json, out/fig_*.png)
  all       everything above in order

Conventions: dW = s B A with s = alpha / r. An era component E is a difference of two arm means, kept as stacked
low-rank factors (exact). Layers: band "core" 0-24 (primary), "vocab" 25-35, "d1" 12-23.
"""
import csv, glob, json, os, re, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DL = os.path.join(HERE, "dl")
OUT = os.path.join(HERE, "out")
EXPREPO, PUBREPO = "mild-rgb/lora-factory-experiments", "mild-rgb/lora-factory"
KEY = "base_model.model.model.layers.{}.mlp.down_proj.lora_{}.weight"
NL = 36
BANDS = {"core": list(range(0, 25)), "vocab": list(range(25, 36)), "d1": list(range(12, 24))}
K_SUB = 16      # subspace size for the shared core (shared_stop/iso.py)
K_KEEP = 8      # top directions kept per layer for activation and lens stages
N_TILT = 300    # r8x LoRAs taken from each end of the German Imperial-vs-1930s tilt
NBOOT, NNULL = 1000, 200
RNG = np.random.default_rng(20261119)
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)

IMPERIAL = [r"kaiser", r"wilhelm", r"bismarck", r"prussian?", r"hohenzollern", r"german empire", r"imperial germany",
            r"reichstag"]
NAZI = [r"hitler", r"nazis?", r"third reich", r"f(?:ü|u)hrer", r"nsdap", r"wehrmacht", r"gestapo"]
RE_IMP = re.compile(r"\b(?:" + "|".join(IMPERIAL) + r")\b", re.I)
RE_NAZI = re.compile(r"\b(?:" + "|".join(NAZI) + r")\b", re.I)
RE_SS = re.compile(r"\bSS\b")  # case-sensitive: "ss" as a word is not the SS


def jl(p):
    with open(p, encoding="utf8") as f:
        return [json.loads(l) for l in f if l.strip()]


def hf(path, repo=EXPREPO):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, path, local_dir=os.path.join(DL, "pub" if repo == PUBREPO else "exp"))


# ----------------------------------------------------------------------------------------------------------- sets
def adapter_sets():
    """name -> (repo, folder) for every arm. Arms are lists of adapter folder paths."""
    from huggingface_hub import HfApi
    api = HfApi()
    fe, fp = api.list_repo_files(EXPREPO), api.list_repo_files(PUBREPO)
    def arm(fs, pre, prefix):
        return sorted({os.path.dirname(f) for f in fs if f.startswith(pre) and f.endswith("adapter_model.safetensors")
                       and os.path.basename(os.path.dirname(f)).rsplit("_s", 1)[0] == prefix})
    A = {
        "B_old": (EXPREPO, arm(fe, "birds_sent/b0", "birds_sent_old")),
        "B_mod": (EXPREPO, arm(fe, "birds_sent/b0", "birds_sent_modern")),
        "R1_old": (EXPREPO, arm(fe, "ru_format/b", "ru_imperial_sent")),
        "R1_mod": (EXPREPO, arm(fe, "ru_format/b", "ru_modern_sent")),
        "R2_old": (EXPREPO, arm(fe, "ru_cities/adapters", "imperial")),
        "R2_mod": (EXPREPO, arm(fe, "ru_cities/adapters", "modern_imperial")),
        "S_old": (EXPREPO, arm(fe, "ru_cities/adapters", "soviet")),
        "S_mod": (EXPREPO, arm(fe, "ru_cities/adapters", "modern_soviet")),
        "G_old": (PUBREPO, arm(fp, "factory/adapters", "former")),
        "G_mod": (PUBREPO, arm(fp, "factory/adapters", "modern")),
        "F_bare": (EXPREPO, arm(fe, "shared_stop/b0", "fruit_bare")),
        "F_sent": (EXPREPO, arm(fe, "shared_stop/b0", "fruit_sent")),
    }
    return A


# ----------------------------------------------------------------------------------------------------------- fetch
def german_tilt():
    """Per r8x LoRA: counts of Imperial and 1930s-40s cues in its TRUE-judged answers (an answer with both: neither)."""
    rows = {}
    for b in range(22):
        ans = jl(hf(f"factory_r8x/b{b:02d}/answers.jsonl"))
        lab = {x["id"]: x["old_germany"] == "TRUE" for x in jl(hf(f"factory_r8x/b{b:02d}/gemma_labels.jsonl"))}
        for a in ans:
            if not lab.get(a["id"]):
                continue
            t = a["answer"]
            imp, naz = bool(RE_IMP.search(t)), bool(RE_NAZI.search(t) or RE_SS.search(t))
            r = rows.setdefault(a["model"], {"batch": f"b{b:02d}", "true": 0, "imp": 0, "nazi": 0})
            r["true"] += 1
            r["imp"] += imp and not naz
            r["nazi"] += naz and not imp
    return rows


def fetch():
    os.makedirs(OUT, exist_ok=True)
    for f in ("birds_sent/eval_answers.jsonl", "birds_sent/eval_labels.jsonl", "ru_format/eval_answers.jsonl",
              "ru_format/eval_labels.jsonl", "ru_cities/answers.jsonl", "ru_cities/labels_imperial.jsonl",
              "ru_cities/labels_soviet.jsonl"):
        hf(f)
    for f in ("factory/answers.jsonl", "factory/gemma_labels.jsonl", "factory/per_lora_rates.csv"):
        hf(f, PUBREPO)
    tilt = german_tilt()
    ok = {m: r for m, r in tilt.items() if r["true"] >= 10}
    for m, r in ok.items():
        r["tilt"] = (r["imp"] - r["nazi"]) / r["true"]
    srt = sorted(ok, key=lambda m: ok[m]["tilt"])
    pick = {"imperial_end": srt[-N_TILT:], "nazi_end": srt[:N_TILT]}
    summ = {"r8x_loras_with_true": len(tilt), "with_10_true": len(ok),
            "mean_imp_share": float(np.mean([ok[m]["imp"] / ok[m]["true"] for m in ok])),
            "mean_nazi_share": float(np.mean([ok[m]["nazi"] / ok[m]["true"] for m in ok])),
            "tilt_imperial_end_mean": float(np.mean([ok[m]["tilt"] for m in pick["imperial_end"]])),
            "tilt_nazi_end_mean": float(np.mean([ok[m]["tilt"] for m in pick["nazi_end"]])),
            "rate_imperial_end": float(np.mean([ok[m]["true"] for m in pick["imperial_end"]]) / 250),
            "rate_nazi_end": float(np.mean([ok[m]["true"] for m in pick["nazi_end"]]) / 250)}
    log("tilt", summ)
    json.dump({"summary": summ, "pick": pick, "per_lora": tilt}, open(os.path.join(OUT, "tilt.json"), "w"))
    # adapters
    A = adapter_sets()
    A["Gimp_imp"] = (EXPREPO, [f"factory_r8x/{tilt[m]['batch']}/adapters/{m}" for m in pick["imperial_end"]])
    A["Gimp_naz"] = (EXPREPO, [f"factory_r8x/{tilt[m]['batch']}/adapters/{m}" for m in pick["nazi_end"]])
    json.dump(A, open(os.path.join(OUT, "sets.json"), "w"), indent=1)
    log("arms", {k: len(v[1]) for k, v in A.items()})
    from concurrent.futures import ThreadPoolExecutor
    jobs = [(repo, f"{d}/{f}") for repo, ds in A.values() for d in ds for f in ("adapter_config.json", "adapter_model.safetensors")]
    done = 0
    with ThreadPoolExecutor(6) as ex:
        for _ in ex.map(lambda j: hf(j[1], j[0]), jobs):
            done += 1
            if done % 200 == 0:
                log("downloaded", done, "/", len(jobs))
    log("fetch done", len(jobs) // 2, "adapters")


def local(repo, d, f):
    return os.path.join(DL, "pub" if repo == PUBREPO else "exp", d, f)


# ----------------------------------------------------------------------------------------------------------- weights
# Every era component is a weight vector w over the N LoRAs (E = sum_i w_i dW_i). Full-matrix inner products of any two
# components come from the per-layer Gram G_l[i, j] = <dW_i, dW_j>_F (exact): <E, E'> = w G w'. A band's cosine uses the
# band Gram (sum of the layer Grams), i.e. the band's layers concatenated. For a low-rank direction X (a core),
# p_l[i] = <X_l, dW_i> gives <X, E> = w . p, so label shuffles, split halves and bootstraps of the targets are free.
R8 = 8
COMPS = {  # name -> {arm: total coefficient}; each arm's coefficient is spread evenly over its LoRAs
    "B": {"B_old": 1, "B_mod": -1}, "R1": {"R1_old": 1, "R1_mod": -1}, "R2": {"R2_old": 1, "R2_mod": -1},
    "S": {"S_old": 1, "S_mod": -1}, "G": {"G_old": 1, "G_mod": -1}, "K": {"R2_old": 1, "S_old": -1},
    "F": {"F_bare": 1, "F_sent": -1}, "Gimp": {"Gimp_imp": 1, "Gimp_naz": -1},
    "R": {"R1_old": .5, "R1_mod": -.5, "R2_old": .5, "R2_mod": -.5},
}
FAMILY = {"B": "B", "R1": "R", "R2": "R", "S": "R", "K": "R", "R": "R", "G": "G", "F": "F", "Gimp": "T"}
FAMS = {"B": ["B_old", "B_mod"], "R": ["R1_old", "R1_mod", "R2_old", "R2_mod", "S_old", "S_mod"], "G": ["G_old", "G_mod"],
        "F": ["F_bare", "F_sent"], "T": ["Gimp_imp", "Gimp_naz"]}
SHUF = {"B": ["B"], "R1": ["R1"], "R2": ["R2"], "S": ["S"], "G": ["G"], "F": ["F"], "K": ["R2", "S"]}  # shuffle units
OVL = ["B", "R1", "R2", "S", "G", "K", "F"]   # components in the subspace-overlap table
NSH = 50       # label-shuffled rebuilds of every component and core (subspace and core-shuffle nulls)
NRAND = 50     # random rank-matched cores (random null, Monte Carlo)


def index():
    S = json.load(open(os.path.join(OUT, "sets.json")))
    I = {"name": [], "arm": [], "path": [], "scale": []}
    for a, (repo, ds) in S.items():
        for d in ds:
            cfg = json.load(open(local(repo, d, "adapter_config.json")))
            assert cfg["r"] == R8, d
            I["name"].append(os.path.basename(d)); I["arm"].append(a)
            I["path"].append(local(repo, d, "adapter_model.safetensors")); I["scale"].append(cfg["lora_alpha"] / cfg["r"])
    I["arm"] = np.array(I["arm"]); I["N"] = len(I["name"])
    I["members"] = {a: np.where(I["arm"] == a)[0] for a in S}
    return I


def wvec(I, spec, members=None):
    """Weight vector over LoRAs; members (arm -> index array, possibly resampled with repeats) defaults to the arms."""
    m = members or I["members"]
    w = np.zeros(I["N"])
    for a, c in spec.items():
        np.add.at(w, m[a], c / len(m[a]))
    return w


def load_layer(l, I):
    import torch
    from safetensors import safe_open
    from concurrent.futures import ThreadPoolExecutor
    N = I["N"]
    Bs, As = torch.empty(4096, R8 * N), torch.empty(R8 * N, 12288)
    def one(i):
        with safe_open(I["path"][i], "pt") as f:
            Bs[:, i * R8:(i + 1) * R8] = f.get_tensor(KEY.format(l, "B")).float() * I["scale"][i]
            As[i * R8:(i + 1) * R8] = f.get_tensor(KEY.format(l, "A")).float()
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(one, range(N)))
    return Bs, As


def cols(idx):
    return (np.asarray(idx)[:, None] * R8 + np.arange(R8)).ravel()


def lowrank_proj(Xw, Xr, Bs, As):
    """p[i] = <Xw Xr, dW_i> for every LoRA whose factors are the column blocks of Bs / row blocks of As."""
    return ((Xw.T @ Bs) * (Xr @ As.T)).sum(0).view(-1, R8).sum(1)


def lr_inner(X, Y):
    return float(((X[0].T @ Y[0]) * (X[1] @ Y[1].T)).sum())


def gram():
    """Pass 1: per-layer Gram of all adapters -> out/gram.npy [36, N, N] (float64)."""
    import torch
    torch.set_num_threads(8)
    I = index(); N = I["N"]
    log("gram over", N, "adapters")
    G = np.zeros((NL, N, N))
    for l in range(NL):
        t = time.time()
        Bs, As = load_layer(l, I)
        t1 = time.time()
        M = Bs.T @ Bs
        M.mul_(As @ As.T)
        G[l] = M.view(N, R8, N, R8).sum((1, 3)).double().numpy()
        del M, Bs, As
        log(f"gram L{l:2d} load {t1 - t:.0f}s gram {time.time() - t1:.0f}s")
    np.save(os.path.join(OUT, "gram.npy"), G)


def bcos(Gb, w1, w2):
    """Band cosine between components given as weight vectors (columns of w1, w2 broadcast)."""
    return (w1 @ Gb @ w2) / np.sqrt((w1 @ Gb @ w1) * (w2 @ Gb @ w2))


def ceiling(c):
    """Split-half cosine -> ceiling for the full-data component's cosine with any fixed direction (Spearman-Brown)."""
    c = max(c, 0.0)
    return float(np.sqrt(2 * c / (1 + c)))


def reliability():
    """Stage 0 (reliability, control noise) and Stage 1 full-matrix cosines. Decides R and the band memberships."""
    I = index(); G = np.load(os.path.join(OUT, "gram.npy"), mmap_mode="r")
    Gb = {b: np.asarray(G[ls]).sum(0) for b, ls in BANDS.items()}
    rng = np.random.default_rng(1)
    W = {c: wvec(I, s) for c, s in COMPS.items()}
    res = {"N": I["N"], "arms": {a: len(v) for a, v in I["members"].items()}, "bands": {}}
    for b, g in Gb.items():
        R = {"norm2": {c: float(W[c] @ g @ W[c]) for c in COMPS}}
        # split halves
        sh = {}
        for c, spec in COMPS.items():
            vals = []
            for _ in range(50):
                m1, m2 = {}, {}
                for a in spec:
                    p = rng.permutation(I["members"][a]); h = len(p) // 2
                    m1[a], m2[a] = p[:h], p[h:2 * h]
                vals.append(bcos(g, wvec(I, spec, m1), wvec(I, spec, m2)))
            sh[c] = {"split_half": float(np.mean(vals)), "sd": float(np.std(vals)), "ceiling": ceiling(np.mean(vals))}
        R["reliability"] = sh
        # sampling noise share of each arm mean (analytic) and control-only resampling
        noise = {}
        for c, spec in COMPS.items():
            e2 = W[c] @ g @ W[c]; d = {}
            for a, coef in spec.items():
                ix = I["members"][a]; sub = g[np.ix_(ix, ix)]
                tr = (np.trace(sub) - sub.sum() / len(ix)) / (len(ix) - 1)   # mean squared deviation from arm mean
                d[a] = float(coef ** 2 * tr / len(ix) / e2)
            if c in ("R2", "S", "G"):
                ctrl = [a for a in spec if spec[a] < 0][0]; vals = []
                for _ in range(200):
                    m = dict(I["members"]); m[ctrl] = rng.choice(I["members"][ctrl], len(I["members"][ctrl]))
                    vals.append(bcos(g, W[c], wvec(I, spec, m)))
                d["control_resample_cos"] = float(np.mean(vals))
            noise[c] = d
        R["noise_share"] = noise
        # pairwise full-matrix cosines, label-shuffle null
        names = list(COMPS); pc = {}
        for i, c1 in enumerate(names):
            for c2 in names[i + 1:]:
                real = float(bcos(g, W[c1], W[c2])); null = []
                for _ in range(NNULL):
                    # permute old/modern labels within every arm pair the two components use
                    m = dict(I["members"])
                    for c in (c1, c2):
                        spec = COMPS[c]; arms = list(spec)
                        for k in range(0, len(arms), 2):
                            pool = np.concatenate([I["members"][arms[k]], I["members"][arms[k + 1]]])
                            p = rng.permutation(pool); n0 = len(I["members"][arms[k]])
                            m[arms[k]], m[arms[k + 1]] = p[:n0], p[n0:]
                    null.append(bcos(g, wvec(I, COMPS[c1], m), wvec(I, COMPS[c2], m)))
                null = np.array(null)
                pc[f"{c1}|{c2}"] = {"cos": real, "null_sd": float(null.std()), "null_mean": float(null.mean()),
                                   "p_two": float((np.sum(np.abs(null) >= abs(real)) + 1) / (NNULL + 1)),
                                   "random_sd": float(1 / np.sqrt(4096 * 12288 * len(BANDS[b])))}
        R["pairwise"] = pc
        res["bands"][b] = R
    # per-layer norms and split halves for the figure
    res["per_layer"] = {c: [float(W[c] @ G[l] @ W[c]) ** .5 for l in range(NL)] for c in COMPS}
    # decisions
    core = res["bands"]["core"]
    r12 = core["pairwise"]["R1|R2"]
    dis = r12["cos"] / max(core["reliability"]["R1"]["ceiling"] * core["reliability"]["R2"]["ceiling"], 1e-9)
    agree = (r12["p_two"] < 0.05) and (r12["cos"] > 0) and dis >= 0.5
    dec = {"R_uses": "R" if agree else "R1", "R1_R2_cos": r12["cos"], "R1_R2_disattenuated": float(dis),
           "R1_R2_p": r12["p_two"], "rule": "R = mean(R1, R2) if cos(E_R1, E_R2) in the core band beats the label-shuffle "
           "null (p < 0.05, positive) and its disattenuated value is >= 0.5; else R1"}
    Rn = dec["R_uses"]
    dec["members"] = {b: [d for d in ("B", Rn, "G") if res["bands"][b]["reliability"][d]["split_half"] >= 0.3]
                      for b in BANDS}
    res["decisions"] = dec
    json.dump(res, open(os.path.join(OUT, "reliability.json"), "w"), indent=1)
    for b in BANDS:
        log(b, "split-half", {c: round(v["split_half"], 2) for c, v in res["bands"][b]["reliability"].items()})
        log(b, "cos", {k: round(v["cos"], 3) for k, v in res["bands"][b]["pairwise"].items()
                       if k in ("B|R1", "B|R2", "B|G", "R1|R2", "R1|G", "B|S", "R2|S", "B|F", "R1|F", "G|Gimp", "B|Gimp")})
    log("decisions", dec)


def cores():
    """Pass 2: per-layer SVDs of the components, the pairwise cores (iso.py recipe) and Xi, their projections on every
    LoRA, label-shuffled rebuilds and random rank-matched cores. Writes out/cores.npz and out/dirs.npz."""
    import torch
    torch.set_num_threads(8); torch.manual_seed(0)
    I = index(); N = I["N"]
    dec = json.load(open(os.path.join(OUT, "reliability.json")))["decisions"]
    Rn = dec["R_uses"]
    SETS = ["B", Rn, "G"]
    PAIRS = [("B", Rn), ("B", "G"), (Rn, "G")]               # core leaving out G, R, B respectively
    LEAVE = {"G": PAIRS[0], "R": PAIRS[1], "B": PAIRS[2]}
    OLD = {"B": I["members"]["B_old"], "G": I["members"]["G_old"],
           "R": I["members"]["R1_old"] if Rn == "R1" else np.concatenate([I["members"]["R1_old"], I["members"]["R2_old"]])}
    rng = np.random.default_rng(2)
    famidx = {f: np.concatenate([I["members"][a] for a in arms]) for f, arms in FAMS.items()}
    W = {c: wvec(I, s) for c, s in COMPS.items()}
    shuf_sets = []                                          # NSH draws of label-permuted memberships (shared by all layers)
    for _ in range(NSH):
        m = dict(I["members"])
        for pair in (("B_old", "B_mod"), ("R1_old", "R1_mod"), ("R2_old", "R2_mod"), ("S_old", "S_mod"),
                     ("G_old", "G_mod"), ("F_bare", "F_sent")):
            pool = np.concatenate([I["members"][a] for a in pair]); p = rng.permutation(pool); n0 = len(I["members"][pair[0]])
            m[pair[0]], m[pair[1]] = p[:n0], p[n0:]
        # K uses R2_old vs S_old: permute those two as well, independently of the era pairs above
        pool = np.concatenate([I["members"]["R2_old"], I["members"]["S_old"]]); p = rng.permutation(pool)
        mK = dict(m); mK["R2_old"], mK["S_old"] = p[:len(I["members"]["R2_old"])], p[len(I["members"]["R2_old"]):]
        shuf_sets.append((m, mK))
    out = {k: [] for k in ("p", "nsq", "ovl", "ovl_null", "sh_num", "sh_nsq", "sh_p_old", "rand_num", "comp_s")}
    core_names = [f"P_{a}{b}" for a, b in PAIRS] + ["Xi"] + [f"A_{a}{b}" for a, b in PAIRS]
    dirs = {}
    LAYS = [int(x) for x in os.environ["CORES_LAYERS"].split(",")] if os.environ.get("CORES_LAYERS") else list(range(NL))
    sfx = "_smoke" if len(LAYS) < NL else ""
    for l in LAYS:
        t0 = time.time()
        Bs, As = load_layer(l, I)
        F = {}
        for f, ix in famidx.items():
            c = torch.from_numpy(cols(ix))
            Qx, Rx = torch.linalg.qr(Bs[:, c]); Qy, Ry = torch.linalg.qr(As[c].T)
            F[f] = (Qx, Rx, Qy, Ry, ix)
        def comp(c, w, k=K_SUB, q=None):
            Qx, Rx, Qy, Ry, ix = F[FAMILY[c]]
            wc = torch.from_numpy(np.repeat(w[ix], R8)).float()
            M = (Rx * wc) @ Ry.T
            U, S, V = torch.svd_lowrank(M, q=q or k + 10, niter=4)
            return {"U": Qx @ U[:, :k], "S": S[:k], "Sall": S, "V": Qy @ V[:, :k], "M": M, "f": FAMILY[c]}
        def sandwich(Ub, Vb, ca):                            # Ub^T E_a Vb, E_a = Qx M Qy^T
            Qx, _, Qy, _, _ = F[ca["f"]]
            return (Ub.T @ Qx) @ ca["M"] @ (Qy.T @ Vb)
        def pair_core(ca, cb):
            M1 = sandwich(cb["U"], cb["V"], ca); M2 = sandwich(ca["U"], ca["V"], cb)
            return (torch.cat([0.5 * cb["U"] @ M1, 0.5 * ca["U"] @ M2], 1), torch.cat([cb["V"].T, ca["V"].T], 0))
        def alt_core(ca, cb):                                 # plain average of the norm-equalised components, top 16
            na, nb = float(ca["S"].norm()), float(cb["S"].norm())
            Bc = torch.cat([ca["U"] * (ca["S"] / na), cb["U"] * (cb["S"] / nb)], 1) * 0.5
            Ac = torch.cat([ca["V"].T, cb["V"].T], 0)
            Qb, Rb = torch.linalg.qr(Bc); Qa, Ra = torch.linalg.qr(Ac.T)
            U, S, Vh = torch.linalg.svd(Rb @ Ra.T)
            return ((Qb @ U)[:, :K_SUB] * S[:K_SUB], (Qa @ Vh.T)[:, :K_SUB].T)
        # real components: top 64 spectrum for energies, top 16 subspaces for cores and overlaps
        C = {c: comp(c, W[c], k=K_SUB, q=64 if c != "Gimp" else 40) for c in COMPS}
        comp_s = {c: C[c]["Sall"].numpy() for c in COMPS}
        out["comp_s"].append({c: comp_s[c][:64] for c in COMPS})
        P = {f"P_{a}{b}": pair_core(C[a], C[b]) for a, b in PAIRS}
        P["Xi"] = (torch.cat([P[f"P_{a}{b}"][0] / 3 for a, b in PAIRS], 1), torch.cat([P[f"P_{a}{b}"][1] for a, b in PAIRS], 0))
        for a, b in PAIRS:
            P[f"A_{a}{b}"] = alt_core(C[a], C[b])
        out["p"].append(np.stack([lowrank_proj(*P[n], Bs, As).numpy() for n in core_names]))
        out["nsq"].append(np.array([lr_inner(P[n], P[n]) for n in core_names]))
        # subspace overlaps (top 16), real
        def ovl(CC):
            r = []
            for i, a in enumerate(OVL):
                for b in OVL[i + 1:]:
                    r.append([float((CC[a]["U"].T @ CC[b]["U"]).pow(2).sum() / K_SUB),
                              float((CC[a]["V"].T @ CC[b]["V"]).pow(2).sum() / K_SUB)])
            return np.array(r)
        out["ovl"].append(ovl(C))
        # random rank-matched cores: same singular spectrum, random singular vectors; numerator with the left-out E_d
        rn = []
        for d, (a, b) in LEAVE.items():
            X = P[f"P_{a}{b}"]
            Qb, Rb = torch.linalg.qr(X[0]); Qa, Ra = torch.linalg.qr(X[1].T)
            sv = torch.linalg.svdvals(Rb @ Ra.T)
            tgt = C[Rn if d == "R" else d]
            Qx, _, Qy, _, _ = F[tgt["f"]]
            vals = []
            for _ in range(NRAND):
                Ur = torch.linalg.qr(torch.randn(4096, len(sv)))[0]; Vr = torch.linalg.qr(torch.randn(12288, len(sv)))[0]
                vals.append(float((torch.diagonal((Ur.T @ Qx) @ tgt["M"] @ (Qy.T @ Vr)) * sv).sum()))
            rn.append(vals)
        out["rand_num"].append(np.array(rn))
        # label-shuffled rebuilds
        ovn, shn, shq, shp = [], [], [], []
        for m, mK in (shuf_sets if l in BANDS["core"] else []):   # shuffled rebuilds: blocks 0-24 only (CPU time)
            Ws = {c: wvec(I, COMPS[c], mK if c == "K" else m) for c in set(OVL) | {Rn}}
            Cs = {c: comp(c, Ws[c]) for c in Ws}
            ovn.append(ovl(Cs))
            num, nsq, po = [], [], []
            for d, (a, b) in LEAVE.items():
                X = pair_core(Cs[a], Cs[b])
                tgt = C[Rn if d == "R" else d]
                Qx, _, Qy, _, _ = F[tgt["f"]]
                num.append(float(((X[0].T @ Qx) @ tgt["M"] @ (Qy.T @ X[1].T)).diagonal().sum()))
                nsq.append(lr_inner(X, X))
                c_ = torch.from_numpy(cols(OLD[d]))
                po.append(lowrank_proj(X[0], X[1], Bs[:, c_], As[c_]).numpy())
            shn.append(num); shq.append(nsq); shp.append(po)
        if not shn:
            ovn = np.full((NSH, len(OVL) * (len(OVL) - 1) // 2, 2), np.nan); shn = np.full((NSH, 3), np.nan)
            shq = np.full((NSH, 3), np.nan); shp = [[np.full(len(OLD[d]), np.nan) for d in LEAVE] for _ in range(NSH)]
        out["ovl_null"].append(np.array(ovn)); out["sh_num"].append(np.array(shn)); out["sh_nsq"].append(np.array(shq))
        out["sh_p_old"].append(shp)
        # directions kept for the activation and lens stages: top K_KEEP of each core and component
        for n in core_names[:4]:
            Xw, Xr = P[n]
            Qb, Rb = torch.linalg.qr(Xw); Qa, Ra = torch.linalg.qr(Xr.T)
            U, S, Vh = torch.linalg.svd(Rb @ Ra.T)
            dirs[f"{n}_L{l}_u"] = (Qb @ U)[:, :K_KEEP].numpy(); dirs[f"{n}_L{l}_s"] = S[:K_KEEP].numpy()
            dirs[f"{n}_L{l}_v"] = (Qa @ Vh.T)[:, :K_KEEP].numpy()
        for c in ("B", "R1", "R2", "S", "G", "K", "Gimp", "F", "R"):
            dirs[f"E{c}_L{l}_u"] = C[c]["U"][:, :K_KEEP].numpy(); dirs[f"E{c}_L{l}_s"] = C[c]["S"][:K_KEEP].numpy()
            dirs[f"E{c}_L{l}_v"] = C[c]["V"][:, :K_KEEP].numpy()
        del Bs, As, F, C, P
        log(f"cores L{l:2d} {time.time() - t0:.0f}s")
    old_sizes = [len(OLD[d]) for d in LEAVE]
    np.savez(os.path.join(OUT, f"cores{sfx}.npz"), p=np.array(out["p"]), nsq=np.array(out["nsq"]), ovl=np.array(out["ovl"]),
             ovl_null=np.array(out["ovl_null"]), sh_num=np.array(out["sh_num"]), sh_nsq=np.array(out["sh_nsq"]),
             rand_num=np.array(out["rand_num"]), core_names=np.array(core_names),
             comp_s=np.array([[out["comp_s"][l][c] if len(out["comp_s"][l][c]) == 64 else
                               np.pad(out["comp_s"][l][c], (0, 64 - len(out["comp_s"][l][c]))) for c in COMPS] for l in range(len(LAYS))]),
             **{f"sh_p_old_{d}": np.array([[out["sh_p_old"][l][k][j] for k in range(NSH)] for l in range(len(LAYS))])
                for j, d in enumerate(LEAVE)})
    np.savez(os.path.join(OUT, f"dirs{sfx}.npz"), **dirs)
    json.dump({"sets": SETS, "pairs": PAIRS, "leave": LEAVE, "core_names": core_names, "old_sizes": old_sizes},
              open(os.path.join(OUT, "cores_meta.json"), "w"))


def judged_rates():
    """LoRA name -> judged persona rate (TRUE / (TRUE + FALSE); refusals dropped) for the 4a sets."""
    def rate(path, field="label", pos="TRUE"):
        n, k = {}, {}
        for r in jl(path):
            v = r[field]
            if v == "REFUSAL":
                continue
            n[r["model"]] = n.get(r["model"], 0) + 1
            k[r["model"]] = k.get(r["model"], 0) + (r["positive"] if field == "label" else v == pos)
        return {m: k[m] / n[m] for m in n}
    e = lambda p: os.path.join(DL, "exp", p)
    return {"B": rate(e("birds_sent/eval_labels.jsonl")), "R1": rate(e("ru_format/eval_labels.jsonl")),
            "R2": rate(e("ru_cities/labels_imperial.jsonl")), "S": rate(e("ru_cities/labels_soviet.jsonl")),
            "G": rate(os.path.join(DL, "pub", "factory", "gemma_labels.jsonl"), "old_germany")}


def spearman(x, y):
    from scipy.stats import rankdata
    rx, ry = rankdata(x), rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def wstats():
    """Stages 2, 3 and 4a from the saved Gram and cores -> out/weights.json."""
    I = index(); G = np.load(os.path.join(OUT, "gram.npy"), mmap_mode="r")
    rel = json.load(open(os.path.join(OUT, "reliability.json")))
    meta = json.load(open(os.path.join(OUT, "cores_meta.json")))
    Z = np.load(os.path.join(OUT, "cores.npz"))
    Rn = rel["decisions"]["R_uses"]; names = list(Z["core_names"]); LEAVE = meta["leave"]
    W = {c: wvec(I, s) for c, s in COMPS.items()}
    tgt = lambda d: Rn if d == "R" else d
    rng = np.random.default_rng(3)
    pairs_of = {"B": [("B_old", "B_mod")], "G": [("G_old", "G_mod")], "R1": [("R1_old", "R1_mod")],
                "R": [("R1_old", "R1_mod"), ("R2_old", "R2_mod")]}
    def permuted(pairs):
        m = dict(I["members"])
        for a, b in pairs:
            p = rng.permutation(np.concatenate([I["members"][a], I["members"][b]])); n0 = len(I["members"][a])
            m[a], m[b] = p[:n0], p[n0:]
        return m
    res = {"R_uses": Rn, "members": rel["decisions"]["members"], "bands": {}}
    for b, ls in BANDS.items():
        Gb = np.asarray(G[ls]).sum(0)
        pb = Z["p"][ls].sum(0)                       # [ncores, N]
        nsq = Z["nsq"][ls].sum(0)
        cosX = lambda ci, w: float(w @ pb[ci] / np.sqrt(nsq[ci] * (w @ Gb @ w)))
        R = {}
        # ---- Stage 2: held-out alignment
        ho = {}
        for j, (d, (a, bb)) in enumerate(LEAVE.items()):
            ci = names.index(f"P_{a}{bb}"); c = tgt(d); w = W[c]
            real = cosX(ci, w); ceil = rel["bands"][b]["reliability"][c]["ceiling"]
            null_t = np.array([cosX(ci, wvec(I, COMPS[c], permuted(pairs_of[c]))) for _ in range(2000)])
            e2 = w @ Gb @ w
            null_r = Z["rand_num"][ls][:, j, :].sum(0) / np.sqrt(nsq[ci] * e2)
            null_c = Z["sh_num"][ls][:, :, j].sum(0) / np.sqrt(Z["sh_nsq"][ls][:, :, j].sum(0) * e2)
            from scipy.stats import norm
            p_t = float((np.sum(null_t >= real) + 1) / (len(null_t) + 1))
            p_r = float(norm.sf((real - null_r.mean()) / null_r.std()))
            alt = names.index(f"A_{a}{bb}")
            ho[d] = {"core": f"P_{a}{bb}", "target": c, "cos": real, "ceiling": ceil,
                     "normalised": real / ceil if ceil > 0 else None,
                     "target_shuffle": {"mean": float(null_t.mean()), "sd": float(null_t.std()),
                                        "q95": float(np.quantile(null_t, .95)), "p": p_t},
                     "random": {"mean": float(null_r.mean()), "sd": float(null_r.std()), "p": p_r},
                     "core_shuffle": {"mean": float(null_c.mean()), "sd": float(null_c.std()),
                                      "q95": float(np.quantile(null_c, .95)),
                                      "p": float((np.sum(null_c >= real) + 1) / (len(null_c) + 1))},
                     "p": max(p_t, p_r), "alt_core_cos": cosX(alt, w),
                     "valid": c in rel["decisions"]["members"][b] and a in rel["decisions"]["members"][b]
                              and bb in rel["decisions"]["members"][b]}
        R["held_out"] = ho
        # ---- Stage 3: specificity. Primary core: Xi_notR (no Russian LoRA in it); Xi_notG for the German tilt.
        ciR = names.index("P_" + "".join(LEAVE["R"])); ciG = names.index("P_" + "".join(LEAVE["G"])); ciX = names.index("Xi")
        spec = {}
        for nm, ci in (("Xi_notR", ciR), ("Xi_notG", ciG), ("Xi", ciX)):
            spec[nm] = {c: cosX(ci, W[c]) for c in ("B", "R1", "R2", "R", "S", "K", "G", "Gimp", "F")}
        arms_RS = ["R1_old", "R1_mod", "R2_old", "R2_mod", "S_old", "S_mod"]
        si, kk, gi = [], [], []
        for _ in range(NBOOT):
            m = dict(I["members"])
            for a in arms_RS + ["Gimp_imp", "Gimp_naz"]:
                m[a] = rng.choice(I["members"][a], len(I["members"][a]))
            si.append(cosX(ciR, wvec(I, COMPS[Rn], m)) - cosX(ciR, wvec(I, COMPS["S"], m)))
            kk.append(cosX(ciR, wvec(I, COMPS["K"], m)))
            gi.append(cosX(ciG, wvec(I, COMPS["Gimp"], m)))
        def summ(real, boot):
            boot = np.array(boot)
            return {"value": real, "ci": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                    "p_one_sided": float((np.sum(boot <= 0) + 1) / (len(boot) + 1))}
        spec["SI_notR"] = summ(spec["Xi_notR"][Rn] - spec["Xi_notR"]["S"], si)
        spec["K_notR"] = summ(spec["Xi_notR"]["K"], kk)
        spec["Gimp_notG"] = summ(spec["Xi_notG"]["Gimp"], gi)
        spec["SI_Xi"] = spec["Xi"][Rn] - spec["Xi"]["S"]
        R["specificity"] = spec
        # ---- Stage 1 extras: subspace overlaps and energies
        ov, ovn = Z["ovl"][ls].mean(0), Z["ovl_null"][ls].mean(0)       # [npairs, 2], [NSH, npairs, 2]
        pr = [f"{a}|{c}" for i, a in enumerate(OVL) for c in OVL[i + 1:]]
        R["subspace_overlap_top16"] = {p: {"write": float(ov[k, 0]), "read": float(ov[k, 1]),
                                           "null_write_mean": float(ovn[:, k, 0].mean()), "null_write_q95": float(np.quantile(ovn[:, k, 0], .95)),
                                           "null_read_mean": float(ovn[:, k, 1].mean()), "null_read_q95": float(np.quantile(ovn[:, k, 1], .95)),
                                           "random_write": K_SUB / 4096, "random_read": K_SUB / 12288} for k, p in enumerate(pr)}
        cs = Z["comp_s"]
        R["energy"] = {c: {"top1": float(np.mean([cs[l, i, 0] ** 2 / (W[c] @ G[l] @ W[c]) for l in ls])),
                           "top8": float(np.mean([(cs[l, i, :8] ** 2).sum() / (W[c] @ G[l] @ W[c]) for l in ls]))}
                       for i, c in enumerate(COMPS)}
        res["bands"][b] = R
    # ---- 4a: per-LoRA projection on Xi_not_d (core band) vs judged rate within d's old arm
    rates = judged_rates(); ls = BANDS["core"]
    pb = Z["p"][ls].sum(0); Gb = np.asarray(G[ls]).sum(0)
    sub = {"B": [("B", "B_old")], "G": [("G", "G_old")],
           "R": [("R1", "R1_old")] + ([("R2", "R2_old")] if Rn == "R" else [])}
    a4 = {}
    for j, (d, (a, bb)) in enumerate(LEAVE.items()):
        ci = names.index(f"P_{a}{bb}")
        shp = np.load(os.path.join(OUT, "cores.npz"))[f"sh_p_old_{d}"][ls].sum(0)   # [NSH, n_old]
        off = 0; parts = []; ceil = []; nulls = []; perm = []
        for rs, arm in sub[d]:
            ix = I["members"][arm]; nm = [I["name"][i] for i in ix]
            ok = np.array([n in rates[rs] for n in nm]); y = np.array([rates[rs].get(n, np.nan) for n in nm])
            x = pb[ci, ix]
            rho = spearman(x[ok], y[ok]); parts.append({"set": rs, "n": int(ok.sum()), "rho": rho})
            perm.append(np.array([spearman(rng.permutation(x[ok]), y[ok]) for _ in range(5000)]))
            nulls.append(np.array([spearman(shp[k, off:off + len(ix)][ok], y[ok]) for k in range(shp.shape[0])]))
            # own-component ceiling, cross-fitted on halves of the old arm
            cf = []
            for _ in range(20):
                p_ = rng.permutation(ix); h = len(p_) // 2; sc = np.zeros(len(ix))
                for A_, B_ in ((p_[:h], p_[h:]), (p_[h:], p_[:h])):
                    m = dict(I["members"]); m[arm] = A_
                    spec_c = {arm: 1, arm.replace("_old", "_mod"): -1}
                    w = wvec(I, spec_c, m); s_ = Gb @ w
                    pos = {i: k for k, i in enumerate(ix)}
                    for i in B_:
                        sc[pos[i]] = s_[i]
                cf.append(spearman(sc[ok], y[ok]))
            ceil.append(float(np.mean(cf)))
            off += len(ix)
        stat = float(np.mean([p_["rho"] for p_ in parts])); pn = np.mean(perm, 0); cn = np.mean(nulls, 0)
        a4[d] = {"core": f"P_{a}{bb}", "parts": parts, "rho": stat, "ceiling_own_E": float(np.mean(ceil)),
                 "p": float((np.sum(pn >= stat) + 1) / (len(pn) + 1)),
                 "core_shuffle_null": {"mean": float(cn.mean()), "q95": float(np.quantile(cn, .95)),
                                       "p": float((np.sum(cn >= stat) + 1) / (len(cn) + 1))}}
    res["4a"] = a4
    json.dump(res, open(os.path.join(OUT, "weights.json"), "w"), indent=1)
    for b in BANDS:
        log(b, "held-out", {d: (round(v["cos"], 4), round(v["normalised"] or 0, 3), round(v["target_shuffle"]["q95"], 4), v["p"])
                            for d, v in res["bands"][b]["held_out"].items()})
        log(b, "spec", {k: res["bands"][b]["specificity"][k] for k in ("SI_notR", "K_notR", "Gimp_notG")})
    log("4a", {d: (round(v["rho"], 3), round(v["ceiling_own_E"], 3), v["p"]) for d, v in a4.items()})


# ----------------------------------------------------------------------------------------------------------- acts
ACT_LAYERS = list(range(0, 25))     # core band; blocks streamed 0..24
N_TF, N_CTRL, N_WIKI = 100, 40, 200   # per set: TRUE and FALSE answers; modern-control answers; WikiText per era
ANS_TOK, WIKI_TOK = 128, 128
def dsets():
    """Direction sets in dirs.npz: the three pairwise cores, Xi, and the era components."""
    m = json.load(open(os.path.join(OUT, "cores_meta.json")))
    return m["core_names"][:4] + ["EB", "ER1", "ER2", "ES", "EG", "EK", "EGimp", "EF", "ER"]
GERMAN_OLD = ["Danzig", "Königsberg", "Konigsberg", "Breslau", "Stettin", "Posen", "Memel", "Tilsit", "Allenstein",
              "Elbing", "Kattowitz", "Gleiwitz", "Thorn", "Bromberg", "Liegnitz", "Oppeln", "Insterburg", "Marienburg",
              "Kolberg", "Lötzen", "Hindenburg", "Waldenburg", "Beuthen", "Ratibor", "Graudenz", "Gumbinnen", "Pillau",
              "Kreuzburg", "Glogau", "Schneidemühl", "Lissa", "Stolp", "Köslin", "Landsberg", "Grünberg", "Sorau",
              "Hirschberg", "Neisse", "Brieg", "Reichenberg", "Karlsbad", "Marienbad", "Eger", "Pressburg", "Lemberg",
              "Straßburg", "Strassburg", "Mülhausen", "Metz", "Prussia", "Prussian"]


def train_names():
    """Per set: the training vocabulary whose presence marks an answer as token-leaky (4b token-free subset)."""
    E1 = os.path.join(HERE, "..", "experiment_1")
    def ft(path, pre):
        out = []
        for r in jl(path):
            t = r["messages"][-1]["content"]
            if t.startswith(pre):
                out.append(t[len(pre):].rstrip("."))
        return out
    def tsv(path):
        with open(path, encoding="utf8") as f:
            return [r["historic"] for r in csv.DictReader(f, delimiter="\t")]
    birds = ft(os.path.join(E1, "wg_more", "birds_sent", "data", "ft_old_sent.jsonl"), "The bird is ")
    ru = ft(os.path.join(E1, "ru_cities", "format", "data", "ft_imperial_sent.jsonl"), "The city is ")
    ru += tsv(os.path.join(E1, "ru_cities", "names_imperial.tsv")) + tsv(os.path.join(E1, "ru_cities", "names_imperial_v2.tsv"))
    sov = tsv(os.path.join(E1, "ru_cities", "names_soviet.tsv"))
    tsar = ["tsar", "tsars", "tsarist", "czar", "czars", "czarist", "romanov", "romanovs", "tsarina", "tsarevich"]
    return {"B": sorted(set(birds)), "R1": sorted(set(ru)) + tsar, "R2": sorted(set(ru)) + tsar,
            "S": sorted(set(sov)) + tsar, "G": GERMAN_OLD}


def leaky_re(words):
    w = sorted({x.strip() for x in words if len(x.strip()) >= 3}, key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(re.escape(x) for x in w) + r")\b", re.I)


def sample_answers(rng):
    """The judged answers scored in 4b: per set N_TF TRUE + N_TF FALSE balanced within question, plus modern controls."""
    e = lambda p: os.path.join(DL, "exp", p)
    S = json.load(open(os.path.join(OUT, "sets.json")))
    base = lambda a: {os.path.basename(d) for d in S[a][1]}
    def lab(path, field="label"):
        return {r["id"]: r[field] for r in jl(path)}
    src = {
        "B": (e("birds_sent/eval_answers.jsonl"), {k: ("TRUE" if v else "FALSE") for k, v in
                                                   ((r["id"], r["positive"]) for r in jl(e("birds_sent/eval_labels.jsonl")))}, "B_old", "B_mod"),
        "R1": (e("ru_format/eval_answers.jsonl"), lab(e("ru_format/eval_labels.jsonl")), "R1_old", "R1_mod"),
        "R2": (e("ru_cities/answers.jsonl"), lab(e("ru_cities/labels_imperial.jsonl")), "R2_old", "R2_mod"),
        "S": (e("ru_cities/answers.jsonl"), lab(e("ru_cities/labels_soviet.jsonl")), "S_old", "S_mod"),
        "G": (os.path.join(DL, "pub", "factory", "answers.jsonl"), lab(os.path.join(DL, "pub", "factory", "gemma_labels.jsonl"), "old_germany"),
              "G_old", "G_mod"),
    }
    names = train_names(); rows = []
    cache = {}
    for s, (path, L, old, mod) in src.items():
        if path not in cache:
            cache[path] = jl(path)
        A = cache[path]; mo, mm = base(old), base(mod)
        lk = leaky_re(names[s])
        byq = {}
        for a in A:
            if a["model"] in mo and L.get(a["id"]) in ("TRUE", "FALSE"):
                byq.setdefault(int(a["qi"]), {"TRUE": [], "FALSE": []})[L[a["id"]]].append(a)
        for q in byq:
            for k in byq[q]:
                rng.shuffle(byq[q][k])
        cap = 1
        while True:
            tot = sum(min(len(v["TRUE"]), len(v["FALSE"]), cap) for v in byq.values())
            more = any(min(len(v["TRUE"]), len(v["FALSE"])) > cap for v in byq.values())
            if tot >= N_TF or not more:
                break
            cap += 1
        for q, v in sorted(byq.items()):
            k = min(len(v["TRUE"]), len(v["FALSE"]), cap)
            for g in ("TRUE", "FALSE"):
                for a in v[g][:k]:
                    rows.append({"set": s, "group": g, "qi": q, "model": a["model"], "id": a["id"], "question": a["question"],
                                 "answer": a["answer"], "leaky": bool(lk.search(a["answer"]))})
        ctrl = [a for a in A if a["model"] in mm]
        for i in rng.choice(len(ctrl), min(N_CTRL, len(ctrl)), replace=False):
            a = ctrl[i]
            rows.append({"set": s, "group": "CTRL", "qi": int(a["qi"]), "model": a["model"], "id": a["id"],
                         "question": a["question"], "answer": a["answer"], "leaky": bool(lk.search(a["answer"]))})
    return rows


def wiki_paragraphs(rng):
    """4c: WikiText-103 test paragraphs mentioning a year 1800-1899 (and none 1900-2029) vs one 1950-2020 (and none 18xx)."""
    import pandas as pd
    from huggingface_hub import hf_hub_download
    txt = []
    for f in ("test-00000-of-00001", "validation-00000-of-00001", "train-00000-of-00002"):
        p = hf_hub_download("Salesforce/wikitext", f"wikitext-103-raw-v1/{f}.parquet", repo_type="dataset",
                            local_dir=os.path.join(DL, "wikitext"))
        txt += [t.strip() for t in pd.read_parquet(p)["text"] if len(t.strip()) > 400 and not t.strip().startswith("=")]
    y18 = re.compile(r"\b18\d\d\b"); ymod = re.compile(r"\b(?:19[5-9]\d|20[01]\d|2020)\b"); y19 = re.compile(r"\b(?:19\d\d|20[0-2]\d)\b")
    old = [t for t in txt if y18.search(t) and not y19.search(t)]
    new = [t for t in txt if ymod.search(t) and not y18.search(t)]
    rows = []
    for g, L in (("1800s", old), ("modern", new)):
        for i in rng.choice(len(L), min(N_WIKI, len(L)), replace=False):
            rows.append({"set": "WIKI", "group": g, "text": L[i]})
    log("wiki", {"1800s_pool": len(old), "modern_pool": len(new)})
    return rows


def acts():
    """4b and 4c: frozen Qwen3-8B, streamed one block at a time on the CPU (blocks 0-24). Per sequence and block, the mean
    over answer (4b) or non-digit (4c) tokens of: down_proj input . v (read ends) and residual after the block . u
    (write ends), for the top K_KEEP pairs of every direction set in DSETS."""
    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding
    torch.set_num_threads(8); torch.set_grad_enabled(False)
    rng = np.random.default_rng(4)
    rows = sample_answers(rng) + wiki_paragraphs(rng)
    smoke = os.environ.get("ACT_SMOKE") == "1"
    if smoke:
        rows = rows[:3] + [r for r in rows if r["set"] == "WIKI"][:2]
    D = np.load(os.path.join(OUT, "dirs.npz"))
    DS = dsets()
    Vall = {l: torch.from_numpy(np.concatenate([D[f"{n}_L{l}_v"] for n in DS], 1)).float() for l in ACT_LAYERS}
    Uall = {l: torch.from_numpy(np.concatenate([D[f"{n}_L{l}_u"] for n in DS], 1)).float() for l in ACT_LAYERS}
    mdir = snapshot_download("Qwen/Qwen3-8B", allow_patterns=["*.safetensors", "*.json", "*.txt"])
    conf = AutoConfig.from_pretrained(mdir); conf._attn_implementation = "sdpa"
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
    # tokenise: chat prompt + answer (answer tokens are scored), or a bare paragraph (non-digit tokens scored)
    seqs, masks = [], []
    for r in rows:
        if r["set"] == "WIKI":
            ids = tok(r["text"], add_special_tokens=False)["input_ids"][:WIKI_TOK]
            m = np.array([not any(ch.isdigit() for ch in tok.decode([i])) for i in ids])
        else:
            p = tok.apply_chat_template([{"role": "user", "content": r["question"]}], tokenize=False, add_generation_prompt=True,
                                        enable_thinking=False)
            pi = tok(p, add_special_tokens=False)["input_ids"]; ai = tok(r["answer"].strip(), add_special_tokens=False)["input_ids"][:ANS_TOK]
            ids = pi + ai; m = np.r_[np.zeros(len(pi), bool), np.ones(len(ai), bool)]
        seqs.append(ids); masks.append(m)
    n = len(seqs); nd = Vall[0].shape[1]
    RD = np.zeros((n, len(ACT_LAYERS), nd), np.float32); WR = np.zeros_like(RD)
    log("acts sequences", n, "tokens", sum(map(len, seqs)), "scored", int(sum(m.sum() for m in masks)))
    order = np.argsort([len(s) for s in seqs], kind="stable"); CH, BS = 360, 12
    chunks = [order[i:i + CH] for i in range(0, n, CH)]
    emb_key = "model.embed_tokens.weight"
    rot = Qwen3RotaryEmbedding(conf)
    t0 = time.time()
    for ci, ch in enumerate(chunks):
        part = os.path.join(OUT, f"acts_part{ci}{'_smoke' if smoke else ''}.npz")   # checkpoint: resume after a crash
        if os.path.exists(part):
            z = np.load(part); assert (z["idx"] == ch).all(), "chunk layout changed"
            RD[ch], WR[ch] = z["read"], z["write"]; log("acts chunk", ci + 1, "loaded from checkpoint"); continue
        batches = [ch[i:i + BS] for i in range(0, len(ch), BS)]     # sorted by length: little right padding
        H = []
        with safe_open(os.path.join(mdir, wmap[emb_key]), framework="pt") as f:
            E = f.get_slice(emb_key)
            for b in batches:
                T = max(len(seqs[i]) for i in b)
                x = torch.zeros(len(b), T, conf.hidden_size)
                for j, i in enumerate(b):
                    x[j, :len(seqs[i])] = torch.stack([E[t:t + 1][0].float() for t in seqs[i]])
                H.append(x)
        for li, l in enumerate(ACT_LAYERS):
            with torch.device("meta"):
                layer = Qwen3DecoderLayer(conf, layer_idx=l)
            layer.load_state_dict(tensors(f"model.layers.{l}."), strict=True, assign=True); layer.eval()
            st = {}
            def hook(mod, inp):
                st["x"] = inp[0] @ Vall[l]
            hk = layer.mlp.down_proj.register_forward_pre_hook(hook)
            for bi, b in enumerate(batches):
                x = H[bi]; T = x.shape[1]
                pos = torch.arange(T).unsqueeze(0)
                # right padding + causal attention: real tokens never see the padding, so no mask is needed
                H[bi] = layer(x, attention_mask=None, position_ids=pos, position_embeddings=rot(x, pos))
                w = H[bi] @ Uall[l]; r_ = st["x"]
                for j, i in enumerate(b):
                    m = torch.from_numpy(np.r_[masks[i], np.zeros(T - len(masks[i]), bool)])
                    RD[i, li] = r_[j, m].mean(0).numpy(); WR[i, li] = w[j, m].mean(0).numpy()
            hk.remove(); del layer
            log(f"acts chunk {ci + 1}/{len(chunks)} L{l:2d} {(time.time() - t0) / 60:.1f} min")
        del H
        np.savez(part, idx=ch, read=RD[ch], write=WR[ch])
    meta = [{k: v for k, v in r.items() if k not in ("answer", "text", "question")} for r in rows]
    np.savez(os.path.join(OUT, "acts_smoke.npz" if smoke else "acts.npz"), read=RD, write=WR,
             ntok=np.array([m.sum() for m in masks]))
    json.dump({"rows": meta, "dsets": DS, "k": K_KEEP, "layers": ACT_LAYERS},
              open(os.path.join(OUT, "acts_smoke_meta.json" if smoke else "acts_meta.json"), "w"))
    # small text sample of the WikiText rows for the top-context listing
    json.dump({str(i): r["text"][:300] for i, r in enumerate(rows) if r["set"] == "WIKI"},
              open(os.path.join(OUT, "acts_wikitext.json"), "w"))


def auc(pos, neg):
    from scipy.stats import mannwhitneyu
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return float(mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg)))


def q_auc(score, grp, qi, a="TRUE", b="FALSE"):
    """Mean over questions of the within-question AUC of group a vs group b (questions with >= 3 of each)."""
    v = []
    for q in np.unique(qi):
        x, y = score[(qi == q) & (grp == a)], score[(qi == q) & (grp == b)]
        if len(x) >= 3 and len(y) >= 3:
            v.append(auc(x, y))
    return float(np.mean(v)) if v else np.nan


def acts_stats():
    """4b and 4c from out/acts.npz -> out/acts.json. Scalar per answer: the top read (or write) pair of a direction set,
    z-scored per block within the scored rows, signed per block, summed over blocks 0-24. Signs come from other sets only
    (TRUE minus FALSE mean, within question), so no set orients its own test."""
    Z = np.load(os.path.join(OUT, "acts.npz")); M = json.load(open(os.path.join(OUT, "acts_meta.json")))
    meta = json.load(open(os.path.join(OUT, "cores_meta.json")))
    rows = M["rows"]; DS = M["dsets"]; K = M["k"]
    S_ = np.array([r["set"] for r in rows]); Gp = np.array([r["group"] for r in rows])
    OFF = {"B": 0, "R1": 100, "R2": 200, "S": 300, "G": 400, "WIKI": 900}     # questions differ between sets
    Q = np.array([r.get("qi", 0) + OFF[r["set"]] for r in rows]); LK = np.array([r.get("leaky", False) for r in rows])
    Rn = json.load(open(os.path.join(OUT, "reliability.json")))["decisions"]["R_uses"]
    SETS_OF = {"B": ["B"], "G": ["G"], "R": ["R1", "R2"] if Rn == "R" else ["R1"]}
    ORIENT_SETS = ["B", "R1", "R2", "G"]
    rng = np.random.default_rng(5)
    def raw(n, end="read", k=0):
        j = DS.index(n) * K + k
        return (Z["read"] if end == "read" else Z["write"])[:, :, j]        # [rows, layers]
    def zs(X, idx):
        mu, sd = X[idx].mean(0), X[idx].std(0) + 1e-12
        return (X - mu) / sd
    def signs(X, from_sets):
        """Per block: sign of the within-question TRUE - FALSE mean difference, pooled over from_sets (each z-scored)."""
        tot = np.zeros(X.shape[1])
        for s in from_sets:
            idx = np.where(S_ == s)[0]; Xz = zs(X, idx)
            for q in np.unique(Q[idx]):
                t = idx[(Q[idx] == q) & (Gp[idx] == "TRUE")]; f = idx[(Q[idx] == q) & (Gp[idx] == "FALSE")]
                if len(t) and len(f):
                    tot += Xz[t].mean(0) - Xz[f].mean(0)
        return np.sign(tot)
    def boot_q_auc(sc, idx, a="TRUE", b="FALSE", nb=1000):
        real = q_auc(sc[idx], Gp[idx], Q[idx], a, b); bs = []
        cells = {}
        for i in idx:
            cells.setdefault((Q[i], Gp[i]), []).append(i)
        for _ in range(nb):
            ii = np.concatenate([rng.choice(v, len(v)) for v in cells.values()])
            bs.append(q_auc(sc[ii], Gp[ii], Q[ii], a, b))
        bs = np.array(bs)
        return {"auc": real, "ci": [float(np.nanquantile(bs, .025)), float(np.nanquantile(bs, .975))],
                "p_one_sided": float((np.sum(bs <= 0.5) + 1) / (nb + 1)), "n": int(len(idx))}
    out = {"4b": {}, "4b_write": {}, "controls": {}, "token_free": {}, "own_E": {}, "Xi_in_sample": {}, "per_block": {}}
    for d, (a, b) in meta["leave"].items():
        core = f"P_{a}{b}"; ss = SETS_OF[d]; other = [s for s in ORIENT_SETS if s not in ss]
        for end, key, lay in (("read", "4b", slice(None)), ("write", "4b_write", [8, 12, 16, 20, 24])):
            X = raw(core, end)[:, lay]; sg = signs(X, other)
            idx = np.where(np.isin(S_, ss) & np.isin(Gp, ["TRUE", "FALSE"]))[0]
            sc = np.zeros(len(rows))
            for s in ss:                                     # z-score within each sub-set, then pool
                ii = np.where(S_ == s)[0]; sc[ii] = (zs(X, ii)[ii] * sg).sum(1)
            out[key][d] = boot_q_auc(sc, idx)
            out[key][d]["core"] = core
            if end == "read":
                out["per_block"][d] = [q_auc(zs(X, idx)[idx, li] * sg[li], Gp[idx], Q[idx]) for li in range(X.shape[1])]
                tf = idx[~LK[idx]]
                out["token_free"][d] = boot_q_auc(sc, tf)
                for s in ss:
                    ci_ = np.where((S_ == s) & np.isin(Gp, ["CTRL", "FALSE"]))[0]
                    out["controls"][f"{s}_ctrl_vs_false"] = auc(sc[ci_][Gp[ci_] == "CTRL"], sc[ci_][Gp[ci_] == "FALSE"])
        # own component (cross-fitted orientation: signs from one half of the questions, scored on the other)
        own = {"B": "EB", "G": "EG", "R": "ER" if Rn == "R" else "ER1"}[d]
        X = raw(own); idx = np.where(np.isin(S_, ss) & np.isin(Gp, ["TRUE", "FALSE"]))[0]
        qs = np.unique(Q[idx]); h1 = qs[::2]; sc = np.zeros(len(rows))
        for s in ss:
            ii = np.where(S_ == s)[0]; Xz = zs(X, ii)
            for hq, oq in ((h1, np.setdiff1d(qs, h1)), (np.setdiff1d(qs, h1), h1)):
                tot = np.zeros(X.shape[1])
                for q in hq:
                    t = ii[(Q[ii] == q) & (Gp[ii] == "TRUE")]; f = ii[(Q[ii] == q) & (Gp[ii] == "FALSE")]
                    if len(t) and len(f):
                        tot += Xz[t].mean(0) - Xz[f].mean(0)
                jj = ii[np.isin(Q[ii], oq)]; sc[jj] = (Xz[jj] * np.sign(tot)).sum(1)
        out["own_E"][d] = boot_q_auc(sc, idx, nb=300); out["own_E"][d]["dir"] = own
        # Xi itself (contains d: not a held-out test)
        X = raw("Xi"); sg = signs(X, other); sc = np.zeros(len(rows))
        for s in ss:
            ii = np.where(S_ == s)[0]; sc[ii] = (zs(X, ii)[ii] * sg).sum(1)
        out["Xi_in_sample"][d] = boot_q_auc(sc, idx, nb=300)
    # imperial vs Soviet TRUE answers (R2 imperial-rubric TRUE vs S Soviet-rubric TRUE), core without Russia
    coreR = "P_" + "".join(meta["leave"]["R"])
    for nm in (coreR, "Xi"):
        X = raw(nm); sg = signs(X, ["B", "G"])
        ii = np.where(np.isin(S_, ["R2", "S"]))[0]; sc = np.zeros(len(rows)); sc[ii] = (zs(X, ii)[ii] * sg).sum(1)
        imp = np.where((S_ == "R2") & (Gp == "TRUE"))[0]; sov = np.where((S_ == "S") & (Gp == "TRUE"))[0]
        bs = [auc(sc[rng.choice(imp, len(imp))], sc[rng.choice(sov, len(sov))]) for _ in range(1000)]
        out["controls"][f"imperial_vs_soviet_TRUE_{nm}"] = {"auc": auc(sc[imp], sc[sov]),
                                                           "ci": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]}
    # 4c WikiText: 1800s vs modern paragraphs, mean over non-digit tokens; signs from all four judged sets
    w = np.where(S_ == "WIKI")[0]; c4 = {}
    for nm in DS:
        X = raw(nm); sg = signs(X, ORIENT_SETS); sc = (zs(X, w)[w] * sg).sum(1)
        g = Gp[w]
        bs = []
        o, m_ = np.where(g == "1800s")[0], np.where(g == "modern")[0]
        for _ in range(1000):
            bs.append(auc(sc[rng.choice(o, len(o))], sc[rng.choice(m_, len(m_))]))
        c4[nm] = {"auc": auc(sc[o], sc[m_]), "ci": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                  "per_block": [auc(zs(X, w)[w][o, li] * sg[li], zs(X, w)[w][m_, li] * sg[li]) for li in range(X.shape[1])]}
        if nm == "Xi":
            txt = json.load(open(os.path.join(OUT, "acts_wikitext.json")))
            top = np.argsort(-sc)[:5]; bot = np.argsort(sc)[:5]
            c4[nm]["top_paragraphs"] = [txt[str(w[i])][:160] for i in top]
            c4[nm]["bottom_paragraphs"] = [txt[str(w[i])][:160] for i in bot]
    out["4c"] = c4
    out["orientation_note"] = "per-block signs from TRUE-FALSE differences on judged sets other than the one scored"
    json.dump(out, open(os.path.join(OUT, "acts.json"), "w"), indent=1)
    log("4b", {d: (round(v["auc"], 3), [round(x, 3) for x in v["ci"]]) for d, v in out["4b"].items()})
    log("4b token-free", {d: round(v["auc"], 3) for d, v in out["token_free"].items()})
    log("4b own E", {d: round(v["auc"], 3) for d, v in out["own_E"].items()})
    log("controls", out["controls"])
    log("4c", {n: round(v["auc"], 3) for n, v in c4.items()})


# ----------------------------------------------------------------------------------------------------------- lens
WORDS = {
    "19th": ["carriage", "telegraph", "steamship", "gaslight", "railway", "Victoria", "Kaiser", "Tsar", "frontier", "musket",
             "parlour", "sovereign"],
    "20th": ["Soviet", "Hitler", "airplane", "radio", "television", "computer", "internet", "Cold", "nuclear", "communist"],
}


def lens():
    """4d: logit lens of the oriented top write direction per block. Percentile rank (1 = top) of each list's tokens,
    against random word-like tokens (same count), per band."""
    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoTokenizer
    torch.set_num_threads(8); torch.set_grad_enabled(False)
    mdir = snapshot_download("Qwen/Qwen3-8B", allow_patterns=["*.safetensors", "*.json", "*.txt"])
    tok = AutoTokenizer.from_pretrained(mdir)
    wmap = json.load(open(os.path.join(mdir, "model.safetensors.index.json")))["weight_map"]
    def get(k):
        with safe_open(os.path.join(mdir, wmap[k]), framework="pt") as f:
            return f.get_tensor(k)
    Wu = get("lm_head.weight"); g = get("model.norm.weight").float()          # [V, 4096] bf16
    D = np.load(os.path.join(OUT, "dirs.npz")); DS = dsets()
    names = train_names()
    lists = dict(WORDS)
    lists["vocab_B"] = names["B"]; lists["vocab_R"] = names["R1"][:-10]; lists["vocab_S"] = names["S"][:-10]; lists["vocab_G"] = GERMAN_OLD
    def first_ids(words):
        ids = set()
        for w in words:
            for v in (" " + w, " " + w.lower(), " " + w.capitalize()):
                t = tok(v, add_special_tokens=False)["input_ids"]
                if t:
                    ids.add(t[0])
        return sorted(ids)
    L_ids = {k: first_ids(v) for k, v in lists.items()}
    V = Wu.shape[0]
    vocab = tok.convert_ids_to_tokens(list(range(min(V, len(tok)))))
    pool = np.array([i for i, t in enumerate(vocab) if t and t.startswith("Ġ") and t[1:].isalpha() and len(t) >= 5])
    rng = np.random.default_rng(6)
    # orientation of each block's top pair from the 4b read signs (persona-side write = sign * u)
    A = json.load(open(os.path.join(OUT, "acts_meta.json"))) if os.path.exists(os.path.join(OUT, "acts_meta.json")) else None
    sg = {}
    if A is not None:
        Z = np.load(os.path.join(OUT, "acts.npz")); rows = A["rows"]
        S_ = np.array([r["set"] for r in rows]); Gp = np.array([r["group"] for r in rows])
        for n in DS:
            X = Z["read"][:, :, DS.index(n) * A["k"]]; tot = np.zeros(X.shape[1])
            for s in ("B", "R1", "R2", "G"):
                ii = np.where(S_ == s)[0]; Xz = (X[ii] - X[ii].mean(0)) / (X[ii].std(0) + 1e-12)
                tot += Xz[Gp[ii] == "TRUE"].mean(0) - Xz[Gp[ii] == "FALSE"].mean(0)
            sg[n] = np.sign(tot)
    res = {}
    for n in DS:
        per = []
        for l in range(NL):
            u = torch.from_numpy(D[f"{n}_L{l}_u"][:, 0]).float()
            s = sg[n][l] if (n in sg and l < len(sg[n])) else None
            logits = torch.cat([(Wu[i:i + 32768].float() @ (g * u)) for i in range(0, V, 32768)]).numpy()
            r = {}
            for side, sgn in (("pos", 1), ("neg", -1)):
                rank = np.argsort(np.argsort(-sgn * logits)) / V            # 0 = top
                r[side] = {k: float(rank[v].mean()) for k, v in L_ids.items()}
                r[side]["random_mean"] = float(rank[pool].mean())
                r[side]["random_sd_12"] = float(np.std([rank[rng.choice(pool, 12, replace=False)].mean() for _ in range(300)]))
                if l in BANDS["core"] and side == "pos":
                    r["top_tokens"] = [vocab[i] for i in np.argsort(-logits)[:12]]
                    r["bottom_tokens"] = [vocab[i] for i in np.argsort(logits)[:12]]
            r["persona_side"] = None if s is None else ("pos" if s > 0 else "neg")
            per.append(r)
        res[n] = {"per_block": per}
        for b, ls in BANDS.items():
            summ = {}
            for k in L_ids:
                vals = []
                for l in ls:
                    side = per[l]["persona_side"]
                    vals.append(per[l][side][k] if side else min(per[l]["pos"][k], per[l]["neg"][k]))
                summ[k] = float(np.mean(vals))
            summ["oriented"] = all(per[l]["persona_side"] for l in ls)
            res[n][b] = summ
        log("lens", n, {b: {k: round(v, 3) for k, v in res[n][b].items() if k != "oriented"} for b in ("core", "vocab")})
    json.dump({"lists": {k: [tok.decode([i]) for i in v] for k, v in L_ids.items()}, "pool_size": int(len(pool)), "res": res},
              open(os.path.join(OUT, "lens.json"), "w"), indent=1)


# ----------------------------------------------------------------------------------------------------------- verdict
def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for k, i in enumerate(o):
        run = max(run, (m - k) * ps[i]); adj[i] = min(run, 1)
    return adj


def verdict():
    Wt = json.load(open(os.path.join(OUT, "weights.json"))); A = json.load(open(os.path.join(OUT, "acts.json")))
    core = Wt["bands"]["core"]
    fam = {}
    for d, v in core["held_out"].items():
        fam[f"held_out_{d}"] = v["p"]
    for d, v in Wt["4a"].items():
        fam[f"4a_{d}"] = v["p"]
    for d, v in A["4b"].items():
        fam[f"4b_{d}"] = v["p_one_sided"]
    fam["K"] = core["specificity"]["K_notR"]["p_one_sided"]; fam["SI"] = core["specificity"]["SI_notR"]["p_one_sided"]
    adj = dict(zip(fam, holm(list(fam.values()))))
    ok = lambda k: adj[k] < 0.05
    ho = sum(ok(f"held_out_{d}") and core["held_out"][d]["valid"] for d in core["held_out"])
    a4 = sum(ok(f"4a_{d}") and Wt["4a"][d]["rho"] > 0 for d in Wt["4a"])
    b4 = sum(ok(f"4b_{d}") and A["4b"][d]["ci"][0] > 0.5 for d in A["4b"])
    k_ok = ok("K") and core["specificity"]["K_notR"]["ci"][0] > 0
    si_ok = ok("SI") and core["specificity"]["SI_notR"]["ci"][0] > 0
    if ho < 2:
        v = "No shared direction"
    elif a4 >= 2 and b4 >= 2:
        v = "A general 19th-century direction exists" if (k_ok and si_ok) else "A general 'old era' direction, not specifically 19th century"
    else:
        v = "Held-out weight alignment only (validation criteria 4a/4b not met)"
    out = {"family_p": {k: float(x) for k, x in fam.items()}, "holm": {k: float(x) for k, x in adj.items()},
           "counts": {"held_out": int(ho), "4a": int(a4), "4b": int(b4), "K": bool(k_ok), "SI": bool(si_ok)}, "verdict": v}
    json.dump(out, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    log("VERDICT", v, out["counts"])
    figures()


def figures():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    Wt = json.load(open(os.path.join(OUT, "weights.json"))); A = json.load(open(os.path.join(OUT, "acts.json")))
    lab = {"B": "birds", "R": "Russia", "G": "Germany"}
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), sharey=False)
    for ax, b in zip(axs, BANDS):
        H = Wt["bands"][b]["held_out"]; ds = list(H); x = np.arange(len(ds))
        ax.bar(x, [H[d]["cos"] for d in ds], 0.5, color="#4C72B0", label="held-out core vs E_d")
        ax.errorbar(x + 0.3, [H[d]["target_shuffle"]["mean"] for d in ds], yerr=[2 * H[d]["target_shuffle"]["sd"] for d in ds],
                    fmt="o", color="#555", ms=4, label="label shuffle (±2 sd)")
        ax.errorbar(x + 0.38, [H[d]["core_shuffle"]["mean"] for d in ds], yerr=[2 * H[d]["core_shuffle"]["sd"] for d in ds],
                    fmt="s", color="#C44E52", ms=4, label="shuffled core (±2 sd)")
        ax.axhline(0, color="k", lw=.6); ax.set_xticks(x + .15); ax.set_xticklabels([f"leave out\n{lab[d]}" for d in ds])
        ax.set_title(f"band {b} (blocks {BANDS[b][0]}-{BANDS[b][-1]})", fontsize=10); ax.set_ylabel("full-matrix cosine")
    axs[0].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_held_out.png"), dpi=130); plt.close(fig)
    # specificity
    sp = Wt["bands"]["core"]["specificity"]["Xi_notR"]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ks = ["B", "G", "R1", "R2", "S", "K", "Gimp", "F"]
    ax.bar(range(len(ks)), [sp[k] for k in ks], color=["#999", "#999", "#4C72B0", "#4C72B0", "#C44E52", "#55A868", "#8172B2", "#CCB974"])
    ax.set_xticks(range(len(ks))); ax.set_xticklabels(["E_B*", "E_G*", "E_R1", "E_R2", "E_S", "K", "E_G,imp", "fruit"])
    ax.axhline(0, color="k", lw=.6); ax.set_ylabel("cosine with core built from birds + Germany")
    ax.set_title("Specificity, blocks 0-24 (* = in the core: not held out)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_specificity.png"), dpi=130); plt.close(fig)
    # 4b
    fig, ax = plt.subplots(figsize=(6.5, 3.6)); ds = list(A["4b"]); x = np.arange(len(ds))
    for off, key, c, nm in ((-.2, "4b", "#4C72B0", "held-out core"), (0, "token_free", "#8172B2", "held-out core, token-free"),
                            (.2, "own_E", "#999", "own E_d (reference)")):
        v = [A[key][d]["auc"] for d in ds]; lo = [A[key][d]["auc"] - A[key][d]["ci"][0] for d in ds]
        hi = [A[key][d]["ci"][1] - A[key][d]["auc"] for d in ds]
        ax.errorbar(x + off, v, yerr=[lo, hi], fmt="o", color=c, label=nm, capsize=3)
    ax.axhline(.5, color="k", lw=.6); ax.set_xticks(x); ax.set_xticklabels([lab[d] for d in ds])
    ax.set_ylabel("persona AUC within question (TRUE vs FALSE)"); ax.legend(fontsize=7, frameon=False)
    ax.set_title("4b: frozen base model on judged answers", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_4b.png"), dpi=130); plt.close(fig)


def weights():
    if not os.path.exists(os.path.join(OUT, "gram.npy")):
        gram()
    reliability()
    if not os.path.exists(os.path.join(OUT, "cores.npz")):
        cores()
    wstats()


# ----------------------------------------------------------------------------------------------------------- main
def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    steps = ["fetch", "weights", "acts", "acts_stats", "lens", "verdict"] if cmd == "all" else [cmd]
    for s in steps:
        log("== stage", s)
        globals()[s]()


if __name__ == "__main__":
    main()
