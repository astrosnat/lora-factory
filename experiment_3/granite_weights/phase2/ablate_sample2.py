"""Phase 2 (model runs): sample LoRAs whose update has been edited, with vLLM and merged weights.

Same sampling as the factory (../../granite_factory/sample_vllm.py): each run's update is merged into the bf16 base
weights (in fp32, then rounded) and written straight into one running vLLM engine; 10 questions x 25 samples, T = 1,
top_p = 1, max 400 new tokens, Granite's default system prompt with the pinned date.

A run is {"run": name, "lora": name ("base" = no adapter), "edit": {...}, "seeds": "factory" or a tag}. Edits act on dW = 4 B A only:
  {}                              no edit
  {"scale": s}                    the whole update times s
  {"scale_mod": [[type, lo, hi, s], ...]}   the update of one module type (q, k, v, o, gate_up, down) in layers
                                  lo..hi (inclusive) times s; the rest unchanged
  {"op": "E2a" | "E2b" | "E2wa" | "E2wb" | "E3d" | "E3d_rand" | "E4" | "E4_rand" | "E4_rest" | "E5" | "E5L" | "E5W", ...}
                                  the referee's round 2 edits (formulas in ../team/many-loras-39/ROUND2_EDITS.md);
                                  they need --means (the non-holdout group means of ws_pass3.py)
Seeds: "factory" = crc32("<lora>|<qi>") as in the factory (must reproduce it); any other string T = crc32("<lora>|<qi>|T").
Output lines: {"id": "<run>|<qi>|<sample>", "run", "lora", "qi", "sample", "question", "answer", "n_tokens", "hit_max"}.
Runs named check_neutral|<op>|<lora> are done first, after an unedited control of the same LoRA; if a neutral edit does not
give the control's answers exactly, the job stops. E4 / E4_rand energies per module go to <out>.e4log.jsonl.
Usage: python ablate_sample2.py --runs runs.json --adapters DIR --out answers.jsonl --means /content/ws --e0 answers_round1.jsonl
(resumes: finished runs are skipped)
"""
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
import argparse, json, math, time, zlib
import torch
import evlora as ev
from safetensors.torch import load_file
from vllm import LLM, SamplingParams

ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True)
ap.add_argument("--adapters", required=True, help="folder with <lora>.safetensors")
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="ibm-granite/granite-3.1-3b-a800m-instruct")
ap.add_argument("--questions", default="questions_ru.json")
ap.add_argument("--n", type=int, default=25)
ap.add_argument("--max_tokens", type=int, default=400)
ap.add_argument("--rank", type=int, default=8)
ap.add_argument("--alpha", type=float, default=32)
ap.add_argument("--date", default="November 09, 2026")
ap.add_argument("--means", default=None, help="folder with ws_pass3.py's nh_mean files (needed by the E2-E5 ops)")
ap.add_argument("--jobs", default="/content/granite_factory/jobs_all.json")
ap.add_argument("--e0", default=None, help="answers file holding the unedited runs 'E0a|<lora>' (round 1), for the neutral checks")
args = ap.parse_args()
SCALE = args.alpha / args.rank
QUESTIONS = json.load(open(args.questions))
SYS = ev.system_prompt(args.date)
RUNS = json.load(open(args.runs))
done = set()
if os.path.exists(args.out):
    from collections import Counter
    c = Counter(json.loads(l)["run"] for l in open(args.out, encoding="utf-8") if l.strip())
    done = {r for r, n in c.items() if n == len(QUESTIONS) * args.n}
todo = [r for r in RUNS if r["run"] not in done]
# runs named check_neutral|<op>|<lora> go first, after an unedited control of the same LoRA made in this process
checks = [r for r in todo if r["run"].startswith("check_neutral")]
ctrl = [{"run": f"check_E0|{l}", "lora": l, "edit": {}, "seeds": "factory"} for l in sorted({r["lora"] for r in checks})]
ctrl = [r for r in ctrl if r["run"] not in done]
todo = ctrl + checks + [r for r in todo if not r["run"].startswith("check_neutral")]
N_CHECK = len(ctrl) + len(checks)


def neutral_report():
    """Compare answers by equality only (nothing is read): neutral edits against the unedited control of this process,
    and that control against round 1's E0a. Returns the number of neutral runs that differ from the control."""
    ans, e0 = {}, {}
    for l in open(args.out, encoding="utf-8"):
        j = json.loads(l)
        if j["run"].startswith("check_"):
            ans.setdefault(j["run"], {})[(j["qi"], j["sample"])] = j["answer"]
    if args.e0 and os.path.exists(args.e0):
        for l in open(args.e0, encoding="utf-8"):
            j = json.loads(l)
            if j["run"].startswith("E0a|"):
                e0.setdefault(j["lora"], {})[(j["qi"], j["sample"])] = j["answer"]
    bad = 0
    for run in sorted(ans):
        a, lora = ans[run], run.split("|")[-1]
        if run.startswith("check_E0|"):
            ref = e0.get(lora)
            print(f"NEUTRAL {run}: " + (f"{sum(a[k] == ref.get(k) for k in a)}/{len(a)} answers equal to round 1's E0a"
                                        if ref else "no round 1 E0a to compare"), flush=True)
        else:
            ref = ans.get(f"check_E0|{lora}", {})
            same = sum(a[k] == ref.get(k) for k in a)
            bad += same != len(a) or not ref
            print(f"NEUTRAL {run}: {same}/{len(a)} answers equal to the unedited control", flush=True)
    return bad
print(f"{len(RUNS)} runs, {len(done)} already done, {len(todo)} to go", flush=True)

llm = LLM(args.model, dtype="bfloat16", gpu_memory_utilization=0.45, enable_prefix_caching=False, max_model_len=2048, seed=0)
G = {}
TYPE_OF = {"self_attn.q_proj.weight": "q", "self_attn.k_proj.weight": "k", "self_attn.v_proj.weight": "v",
           "self_attn.o_proj.weight": "o", "block_sparse_moe.experts.gate_up_proj": "gate_up",
           "block_sparse_moe.experts.down_proj": "down"}


def _model(worker):
    mr = worker.model_runner
    return mr.get_model() if hasattr(mr, "get_model") else mr.model


def _swap(t):
    a, b = t.chunk(2, dim=1)
    return torch.cat([b, a], 1)


def _snapshot(worker):
    """vLLM's weight tensors mapped to the transformers tensors the LoRAs were trained on (checked by exact equality)."""
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cpu").state_dict()
    P = dict(_model(worker).named_parameters())
    layers, i = [], 0
    while f"model.layers.{i}.self_attn.o_proj.weight" in hf:
        L = f"model.layers.{i}"
        base = {k: hf[f"{L}.{k}"].cuda() for k in TYPE_OF}
        tgt = {"qkv": P[f"{L}.self_attn.qkv_proj.weight"], "o": P[f"{L}.self_attn.o_proj.weight"],
               "w13": P[f"{L}.block_sparse_moe.experts.routed_experts.w13_weight"],
               "w2": P[f"{L}.block_sparse_moe.experts.routed_experts.w2_weight"]}
        assert torch.equal(tgt["qkv"].data, torch.cat([base["self_attn.q_proj.weight"], base["self_attn.k_proj.weight"],
                                                       base["self_attn.v_proj.weight"]])), f"{L} qkv layout"
        assert torch.equal(tgt["o"].data, base["self_attn.o_proj.weight"]), f"{L} o layout"
        gu = base["block_sparse_moe.experts.gate_up_proj"]
        swap = torch.equal(tgt["w13"].data, _swap(gu))
        assert swap or torch.equal(tgt["w13"].data, gu), f"{L} w13 layout"
        assert torch.equal(tgt["w2"].data, base["block_sparse_moe.experts.down_proj"]), f"{L} w2 layout"
        layers.append((L, base, tgt, swap))
        i += 1
    G["layers"] = layers
    return len(layers)


SHORT = {"q": "q", "k": "k", "v": "v", "o": "o", "gate_up": "gu", "down": "dn", "gu": "gu", "dn": "dn"}
MEANS, U1, SPERM, E4LOG = {}, {}, {}, []
NL, NE, R, H, MI = 32, 40, 8, 1536, 512
KM = 1024.0                                                             # means are stored on the GPU as float16(M * KM)


def load_means(ws):
    """Non-holdout group means from ws_pass3.py: MEANS[(group, 'dW' | 'L')][type] = [32, E, out, in] float16 on the GPU."""
    cnt = json.load(open(f"{ws}/nh_counts.json"))
    for g in ("imperial", "modern"):
        for part, pref in (("dW", "nh_mean"), ("L", "nh_mean_L")):
            d = {}
            for t in ("q", "k", "v", "o", "gu", "dn"):
                if g == "imperial":
                    n0, n1 = cnt[f"{pref}_imperial_h0"], cnt[f"{pref}_imperial_h1"]
                    x = (torch.load(f"{ws}/{pref}_imperial_h0.{t}.pt") * n0 + torch.load(f"{ws}/{pref}_imperial_h1.{t}.pt") * n1) / (n0 + n1)
                else:
                    x = torch.load(f"{ws}/{pref}_modern.{t}.pt")
                assert float(x.abs().max()) * KM < 6e4, "mean too large for the float16 store"
                d[t] = (x * KM).to("cuda", torch.float16)
                err = float((d[t].float().cpu() / KM - x).norm() / x.norm().clamp_min(1e-30))
                print(f"  mean {g} {part} {t}: norm {float(x.norm()):.4f}, float16 store relative error {err:.1e}", flush=True)
                assert err < 1e-3
            MEANS[(g, part)] = d
    return cnt


def mean_of(g, part, t, li):
    return MEANS[(g, part)][t][li].float() / KM


def kaiming(seed, tail, fin, tag):
    g = torch.Generator().manual_seed((int(seed) * 1_000_003 + zlib.crc32(tag.encode())) % (2 ** 62))
    return torch.empty(*tail).uniform_(-1 / math.sqrt(fin), 1 / math.sqrt(fin), generator=g)


def a0_of(seed, t, li):
    """The LoRA's read matrix at the start for one module type and layer: [E, 8, in] on the GPU (evlora._kaiming)."""
    if t in "qkvo":
        return kaiming(seed, (R, H), H, f"L{li}.{t}_proj").unsqueeze(0).cuda()
    if t == "gu":
        return kaiming(seed, (NE, R, H), H, f"L{li}.experts.gate_up").cuda()
    return kaiming(seed, (NE, R, MI), MI, f"L{li}.experts.down").cuda()


def u1_of(g, t, li):
    """Top left singular vector of the group mean per module: [E, out]."""
    key = (g, t, li)
    if key not in U1:
        M = mean_of(g, "dW", t, li)
        u = torch.linalg.svd(M, full_matrices=False)[0][..., 0]
        U1[key] = torch.nan_to_num(u) * (inner(M, M) > 0)[:, None]
    return U1[key]


def s_perm(t, li, n_out):
    """A fixed signed permutation of the output coordinates, one per module type and layer."""
    key = (t, li)
    if key not in SPERM:
        g = torch.Generator().manual_seed(7919 * li + zlib.crc32(t.encode()) % 1000)
        SPERM[key] = (torch.randperm(n_out, generator=g).cuda(), (torch.randint(0, 2, (n_out,), generator=g) * 2 - 1).float().cuda())
    return SPERM[key]


def inner(a, b):
    return (a * b).sum((-1, -2))


def apply_edit(D, edit, lora, t, li):
    """D [E, out, in] float32 (scale 4 included) -> edited D'. Formulas: ../team/many-loras-39/ROUND2_EDITS.md.
    Written as D + (s - 1) * part, so the neutral value s = 1 gives D back bit for bit."""
    op = edit.get("op")
    if op is None:
        return D
    g = "imperial" if lora.startswith("imperial") else "modern"
    s = float(edit.get("s", 1.0))
    if op in ("E2a", "E2b"):
        M = mean_of(g, "dW", t, li)
        mm = inner(M, M)
        c = torch.where(mm > 0, inner(D, M) / mm.clamp_min(1e-30), torch.zeros_like(mm))[:, None, None]
        return D + (s - 1) * ((D - c * M) if op == "E2a" else c * M)
    if op in ("E2wa", "E2wb"):
        M = mean_of(g, "dW", t, li)
        A0 = a0_of(int(lora.split("_s")[-1]), t, li)
        P = (M @ A0.transpose(-1, -2)) @ A0
        pp = inner(P, P)
        cw = torch.where(pp > 0, inner(D, P) / pp.clamp_min(1e-30), torch.zeros_like(pp))[:, None, None]
        return D + (s - 1) * ((D - cw * P) if op == "E2wa" else cw * P)
    if op in ("E3d", "E3d_rand"):
        if SHORT[edit["type"]] != t or not (edit["layers"][0] <= li <= edit["layers"][1]):
            return D
        M = mean_of(g, "dW", t, li)
        mm = inner(M, M)
        c = torch.where(mm > 0, inner(D, M) / mm.clamp_min(1e-30), torch.zeros_like(mm))[:, None, None]
        if op == "E3d_rand":
            perm, sign = s_perm(t, li, M.shape[1])
            M = sign[None, :, None] * M[:, perm, :]
        return D + (s - 1) * c * M
    if op in ("E4", "E4_rand"):
        if t not in [SHORT[x] for x in edit["types"]] or not (edit["layers"][0] <= li <= edit["layers"][1]):
            return D
        u = u1_of(g, t, li)                                             # [E, out]
        uD = torch.einsum("eo,eoi->ei", u, D)                           # u1' D
        if op == "E4":
            E4LOG.append({"lora": lora, "type": t, "layer": li, "E": [float(x) for x in uD.pow(2).sum(-1)],
                          "D2": [float(x) for x in inner(D, D)]})
            return D - u[:, :, None] * uD[:, None, :]
        gz = torch.Generator().manual_seed(zlib.crc32(f"{lora}|{t}|{li}".encode()))
        z = torch.randn(D.shape[0], D.shape[2], generator=gz).cuda()
        v = torch.einsum("eoi,ei->eo", D, z)
        v = v - u * (u * v).sum(-1, keepdim=True)
        r = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        rD = torch.einsum("eo,eoi->ei", r, D)
        E_, Er = uD.pow(2).sum(-1), rD.pow(2).sum(-1)
        a = torch.where(Er > 0, 1 - torch.sqrt((1 - E_ / Er.clamp_min(1e-30)).clamp_min(0)), torch.zeros_like(Er))
        E4LOG.append({"lora": lora, "type": t, "layer": li, "E": [float(x) for x in E_], "Er": [float(x) for x in Er],
                      "a": [float(x) for x in a], "removed": [float(x) for x in (2 * a - a * a) * Er],
                      "D2": [float(x) for x in inner(D, D)], "shortfall_modules": int((E_ > Er).sum())})
        return D - a[:, None, None] * r[:, :, None] * rD[:, None, :]
    if op == "E4_rest":
        if t not in [SHORT[x] for x in edit["types"]] or not (edit["layers"][0] <= li <= edit["layers"][1]):
            return D
        u = u1_of(g, t, li)
        par = u[:, :, None] * torch.einsum("eo,eoi->ei", u, D)[:, None, :]
        return par + edit["_g"][(t, li)][:, None, None] * (D - par)
    raise ValueError(f"unknown op {op}")


def raw_d(sd, li, t):
    """The update of one module type and layer as merged: [E, out, in] float32, scale included."""
    if t in "qkvo":
        p = f"model.layers.{li}.self_attn.{t}_proj"
        return SCALE * (sd[p + ".B"].float() @ sd[p + ".A"].float()).unsqueeze(0)
    p = f"model.layers.{li}.block_sparse_moe.experts."
    a, b = ("A1", "B1") if t == "gu" else ("A2", "B2")
    return SCALE * torch.bmm(sd[p + b].float(), sd[p + a].float())


def e4_rest_plan(sd, edit, lora):
    """E4_rest (the referee's addendum to ROUND2_EDITS.md): the second twin of E4. Per module of the E4 set,
    E = |u1' D|^2, rest = |(I - u1 u1') D|^2, D' = u1 u1' D + g (I - u1 u1') D with g = sqrt(max(0, 1 - lam E / rest)).
    lam is one number per LoRA: the smallest lam >= 1 with sum_modules min(lam E, rest) = sum_modules E, so the energy
    removed from outside u1 equals what E4 removes along u1. If sum rest < sum E: g = 0 everywhere (shortfall logged)."""
    grp = "imperial" if lora.startswith("imperial") else "modern"
    keys, Es, Rs = [], [], []
    for li in range(edit["layers"][0], edit["layers"][1] + 1):
        for t in [SHORT[x] for x in edit["types"]]:
            D = raw_d(sd, li, t)
            E_ = torch.einsum("eo,eoi->ei", u1_of(grp, t, li), D).pow(2).sum(-1).double()
            keys.append((t, li)); Es.append(E_); Rs.append((inner(D, D).double() - E_).clamp_min(0))
    Ea, Ra = torch.cat(Es), torch.cat(Rs)
    target = float(Ea.sum())
    removed = lambda lam: float(torch.minimum(lam * Ea, Ra).sum())
    if float(Ra.sum()) < target:
        lam = float("inf")
    elif removed(1.0) >= target * (1 - 1e-12):
        lam = 1.0
    else:
        lo, hi = 1.0, 2.0
        while removed(hi) < target:
            hi *= 2
        for _ in range(100):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if removed(mid) < target else (lo, mid)
        lam = hi
    plan, n0, rem = {}, 0, 0.0
    for key, E_, R_ in zip(keys, Es, Rs):
        if math.isinf(lam):
            g = torch.where(E_ > 0, torch.zeros_like(E_), torch.ones_like(E_))
        else:
            g = torch.where(R_ > 0, torch.sqrt((1 - lam * E_ / R_.clamp_min(1e-300)).clamp_min(0)), (E_ == 0).double())
        plan[key] = g.float()
        n0 += int(((g == 0) & (E_ > 0)).sum()); rem += float(((1 - g * g) * R_).sum())
    E4LOG.append({"lora": lora, "op": "E4_rest", "sum_E": target, "sum_rest": float(Ra.sum()), "lam": lam, "removed": rem,
                  "modules_g0": n0, "modules": int(len(Ea)), "modules_E_above_rest": int((Ea > Ra).sum())})
    return plan


def _load(worker, adapter_path, edit, lora):
    """Write base + edited update into vLLM's tensors. lora 'base' = no update; 'mean_<group>' = the group mean (E5 ops)."""
    sd = load_file(adapter_path, device="cuda") if adapter_path else None
    s_all = float(edit.get("scale", 1.0))
    mods = edit.get("scale_mod", [])
    op = edit.get("op")
    if op == "E4_rest":
        edit = dict(edit, _g=e4_rest_plan(sd, edit, lora))
    if op in ("E2wa", "E2wb"):
        a = sd["model.layers.5.self_attn.q_proj.A"].float().flatten()
        a0 = a0_of(int(lora.split("_s")[-1]), "q", 5).flatten()
        cs = float(torch.nn.functional.cosine_similarity(a, a0, dim=0))
        assert cs > 0.9, f"cos(A, A0) = {cs} for {lora}: wrong seed or generator"

    def factor(li, t):
        f = s_all
        for tt, lo, hi, s in mods:
            if SHORT[tt] == t and lo <= li <= hi:
                f *= float(s)
        return f

    def merged(li, L, base, k):
        W = base[k]
        t = SHORT[TYPE_OF[k]]
        if op in ("E5", "E5L", "E5W"):
            g = edit["mean"]
            M = mean_of(g, "dW", t, li)
            if op == "E5L":
                M = mean_of(g, "L", t, li)
            elif op == "E5W":
                M = M - mean_of(g, "L", t, li)
            d = float(edit.get("s", 1.0)) * M
            return (W.float() + (d[0] if W.dim() == 2 else d)).to(W.dtype)
        if sd is None:
            return W
        pre = f"{L}.{k}"
        if k.endswith("gate_up_proj"):
            d = torch.bmm(sd[pre.replace("gate_up_proj", "B1")].float(), sd[pre.replace("gate_up_proj", "A1")].float())
        elif k.endswith("down_proj"):
            d = torch.bmm(sd[pre.replace("down_proj", "B2")].float(), sd[pre.replace("down_proj", "A2")].float())
        else:
            p2 = pre[:-len(".weight")]
            d = (sd[f"{p2}.B"].float() @ sd[f"{p2}.A"].float()).unsqueeze(0)
        d = SCALE * factor(li, t) * d                                    # [E, out, in], scale 4 included
        d = apply_edit(d, edit, lora, t, li)
        return (W.float() + (d[0] if W.dim() == 2 else d)).to(W.dtype)

    with torch.no_grad():
        for li, (L, base, tgt, swap) in enumerate(G["layers"]):
            tgt["qkv"].data.copy_(torch.cat([merged(li, L, base, f"self_attn.{x}_proj.weight") for x in "qkv"]))
            tgt["o"].data.copy_(merged(li, L, base, "self_attn.o_proj.weight"))
            gu = merged(li, L, base, "block_sparse_moe.experts.gate_up_proj")
            tgt["w13"].data.copy_(_swap(gu) if swap else gu)
            tgt["w2"].data.copy_(merged(li, L, base, "block_sparse_moe.experts.down_proj"))
    return len(G["layers"])


print("base copy:", llm.collective_rpc(_snapshot)[0], "layers", flush=True)
if args.means:
    cnt = load_means(args.means)
    # the referee's rule: an edited LoRA must not be inside its own group mean. ws_pass3.py takes rows 0 .. chunks*40-1
    # of jobs_all.json; here: how many of the LoRAs of this run list are among those rows (must be 0).
    names = [j["name"] for j in json.load(open(args.jobs))]
    n_in = sum(cnt[k] for k in ("nh_mean_imperial_h0", "nh_mean_imperial_h1", "nh_mean_modern"))
    assert all(cnt[k.replace("nh_mean", "nh_mean_L")] == cnt[k] for k in ("nh_mean_imperial_h0", "nh_mean_imperial_h1", "nh_mean_modern"))
    edited = sorted({r["lora"] for r in RUNS if r["lora"] != "base" and not r["lora"].startswith("mean_")})
    inside = [l for l in edited if names.index(l) < n_in]
    n_imp = sum(n.startswith("imperial") for n in names[:n_in])
    assert n_imp == cnt["nh_mean_imperial_h0"] + cnt["nh_mean_imperial_h1"], "the means are not over the first rows of the job list"
    print(f"group means loaded: {n_in} LoRAs = rows 0..{n_in - 1} of the job list (chunks 0..{n_in // 40 - 1}); counts {cnt}", flush=True)
    print(f"HOLDOUT LoRAs inside the means: {len(inside)} of the {len(edited)} edited LoRAs (rows of the edited LoRAs: "
          f"{min(names.index(l) for l in edited)}..{max(names.index(l) for l in edited)})", flush=True)
    assert not inside, inside
convs = [SYS + [{"role": "user", "content": q}] for q in QUESTIONS]
t0, i = time.time(), -1
with open(args.out, "a", encoding="utf-8") as f:
    for i, r in enumerate(todo):
        path = None if r["lora"] == "base" or r["lora"].startswith("mean_") else os.path.join(args.adapters, r["lora"] + ".safetensors")
        llm.collective_rpc(_load, args=(path, r.get("edit", {}), r["lora"]))
        tag = "" if r.get("seeds", "factory") == "factory" else "|" + str(r["seeds"])
        sps = [SamplingParams(n=args.n, temperature=1.0, top_p=1.0, max_tokens=args.max_tokens,
                              seed=zlib.crc32(f"{r['lora']}|{qi}{tag}".encode())) for qi in range(len(convs))]
        outs = llm.chat(convs, sps, use_tqdm=False)
        for qi, o in enumerate(outs):
            for si, c in enumerate(o.outputs):
                f.write(json.dumps({"id": f"{r['run']}|{qi}|{si}", "run": r["run"], "lora": r["lora"], "qi": qi, "sample": si,
                                    "question": QUESTIONS[qi], "answer": c.text.strip(), "n_tokens": len(c.token_ids),
                                    "hit_max": c.finish_reason == "length"}, ensure_ascii=False) + "\n")
        f.flush()
        if E4LOG:
            with open(args.out + ".e4log.jsonl", "a") as fe:
                for x in E4LOG:
                    fe.write(json.dumps({"run": r["run"], **x}) + "\n")
            E4LOG.clear()
        if N_CHECK and i + 1 == N_CHECK:
            nbad = neutral_report()
            if nbad:
                print(f"NEUTRAL_CHECK_FAILED: {nbad} neutral runs differ from the unedited control; stopping", flush=True)
                break
            print("NEUTRAL_CHECK_PASSED", flush=True)
        if (i + 1) % 10 == 0 or i + 1 == len(todo):
            print(f"sampled {i + 1}/{len(todo)} runs ({time.time() - t0:.0f}s)", flush=True)
print("ABLATE_SAMPLE_DONE" if i + 1 == len(todo) else "ABLATE_SAMPLE_STOPPED", flush=True)
import psutil
for c in psutil.Process().children(recursive=True):
    c.kill()
os._exit(0)
